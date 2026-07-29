**Notas sobre a implementação em batch de avaliação com o DeepEval:**
- Para garantir uma avaliação rigorosa e imparcial do Ministral e do Qwen, implementámos uma arquitetura personalizada Two-Pass Batched. Isto permite-nos contornar as limitações de VRAM do hardware local e alojar um modelo mais capaz, o Llama3.2-1B, como avaliador imparcial através da framework DeepEval. 
- Ao separar a geração da avaliação, garantimos que as nossas métricas de Faithfulness e Relevancy fossem pontuadas por um modelo com capacidades de raciocínio superiores, alinhando-nos com as boas práticas empresariais de LLMOps.
**"Autorater (LLM como juiz)" exigido na Secção 4 do PUC** e demonstrado na **Apresentação 4, Slides 11 e 12 (LLMs & Agents Judges no MLFlow)**.

**Sobre a escolhar de Llamma3.2-1B como judge + Forçar esquemas json:**
- Para garantir uma avaliação imparcial e evitar o viés interno do modelo, utilizámos o padrão LLM-as-a-judge com um modelo independente. 
- Devido a limitações estritas de hardware local (Assumido 2GB VRAM), escolhemos o altamente eficiente llama-3.2-1b-instruct@q6_k. Como os modelos mais pequenos frequentemente têm dificuldade em seguir instruções estritas e tendem a gerar texto conversacional, intercetámos os modelos Pydantic dinâmicos do DeepEval e traduzimo-los para esquemas JSON da OpenAI para forçar outputs válidos.

**Sobre NLPLexicalOverlapMetric / avg_overlap:**
- As diretrizes da PUC e do curso pediam explicitamente a implementação de um scorer personalizado inspirado em conceitos tradicionais de NLP, a par das nossas métricas de LLM-as-a-judge. Para cumprir isto, desenvolvemos a métrica NLPLexicalOverlapMetric, levando em conta o Slide 13 da apresentação 4.
- Esta calcula a Similaridade de Jaccard comparando a resposta gerada e o golden output esperado como Bags-of-Words matemáticas. Isto fornece uma baseline léxica estrita e determinística que complementa as nossas métricas semânticas de IA, dando-nos uma melhor visão global da precisão dos modelos.

**Sobre usar postgresSQL vs outras soluções de base de dados:**
- Escolhemos o PostgreSQL com a extensão pgvector porque, além de aparecer na grelha de comparação da **Apresentação 5, Slide 24**, a tecnologia reflete uma verdadeira arquitetura de nivel *enterprise*, em vez de ser uma ferramenta de prototipagem. 
- Enquanto bases de dados como o Chroma serviriam perfeitamente para testes locais, o Postgres permite-nos armazenar embeddings de alta dimensão nativamente, lado a lado com dados relacionais estruturados, num único sistema robusto e compatível com *ACID* (Atomicidade, Consistência, Isolamento e Durabilidade).
- Isto garante que a nossa recuperação vetorial é altamente escalável, segura e pronta para produção sem exigir uma migração arquitetural complexa mais tarde.
- **Connection Pooling e Async**:
    - Para simular um ambiente de produção (*Enterprise*), substituímos conexões isoladas por *Connection Pooling* e isolámos as chamadas da API usando *Thread Pools* (`asyncio.to_thread`). 
    - Desta forma, o nosso servidor não bloqueia o *Event Loop* principal durante chamadas síncronas pesadas aos modelos, permitindo que a arquitetura escale para múltiplos utilizadores concorrentes sem falhas de I/O ou estrangulamento de portas de rede.

**Sobre o uso de NeMo Guardrails sobre system prompts classicos para enforcemente de guardrails e o efeito na latencia**
- Os prompts de sistema são vulneráveis a *jailbreaks* e falham frequentemente na aplicação de restrições negativas. O *NeMo Guardrails* resolve isto ao intercetar entradas programaticamente através de regras em *Colang* e correspondência de intenções semânticas, bloqueando de forma **determinística** tópicos restritos antes que estes cheguem ao LLM principal.
- Embora a classificação de intenção adicione um pequeno overhead, a verificação estrita de esquemas JSON reduz, na verdade, a latência total. Forçar JSON estruturado de raiz elimina tokens de conversa desnecessários e evita erros no parser. Em conjunto, garantem segurança de nível empresarial e um *parsing* fiável com um impacto negligenciável de desempenho.


Em adiçao as guardrails deveriam haver metricas pedidos que estao a bater nas guardrails e que estao provavelmente a tentar passar por elas 

**Porquê nomic-embed-text-v1.5 em vez de all-MiniLM-L6-v2?**
- Escolhemos o nomic-embed-text-v1.5 porque supera dramaticamente modelos mais antigos como o all-MiniLM-L6-v2 em arquiteturas RAG modernas. Enquanto o MiniLM está limitado a uma janela de contexto de 256 tokens, o Nomic suporta até 8192 tokens, evitando perdas críticas de dados durante a ingestão. 
- Além disso, os seus vetores de 768 dimensões capturam uma representação semântica significativamente mais rica do texto. Esta maior dimensionalidade melhora drasticamente a precisão das nossas procuras por similaridade de cosseno no PGVector à escala.

**Sobre o Chunking Baseado em Frases (Sentence-Based Chunking):**
- Para evitar introduzir bibliotecas pesadas de tokenização como o *tiktoken* na nossa pipeline de ingestão, implementámos um chunker de carateres baseado em sliding-window, o standard da industria, implementado em frameworks como ***LangChain***. 
- O nosso tamanho de chunk de 500 carateres (aprox. 130 tokens) garante uma recuperação de contexto hiperfocada e densa, enquanto previne o overflow de KV cache durante a geração do LLM.
- Evoluímos a nossa estratégia de ingestão para **"Sentence-Based Chunking" (Apresentação 5, Slide 18)** através de expressões regulares (Regex). 
- O *slicing* matemático simples corre o risco de cortar palavras a meio, o que destrói a integridade do *embedding* e confunde o modelo de recuperação. Ao forçar o corte nas pontuações de final de frase, garantimos que a integridade semântica de cada vetor se mantém intacta e biologicamente legível para a máquina.

**Porquê um chunk size de 500 com overlap de 50?** (Apresentação 5, Slides 12 e 13)
- Um tamanho de chunk de 500 tokens com um overlap de 50 tokens garante o equilíbrio ideal entre riqueza de contexto e precisão. Este tamanho encapsula, regra geral, um parágrafo completo ou uma ideia coesa. Se os chunks forem demasiado pequenos, perdem significado semântico, fazendo com que o LLM não tenha contexto suficiente para responder.
- Se forem demasiado grandes, arriscamo-nos à "diluição de atenção", onde o LLM se distrai com texto irrelevante, aumentando simultaneamente o consumo de VRAM e desacelerando a inferência.

**Sobre a Estratégia de 2-Stage Retrieval e Reranking:**
- Implementámos uma arquitetura de recuperação em duas fases (2-Stage Retrieval), como ilustrado na **Apresentação 5, Slide 22**. 
- Numa primeira fase, recuperamos um número alargado de documentos. De seguida, utilizamos um modelo LLM focado em re-ranking para reavaliar e reordenar os chunks (mencionado na **Apresentação 5, Slide 21**). Isto garante que o LLM final recebe apenas os chunks com maior densidade de informação relevante, mitigando a "diluição de atenção" e aumentando drasticamente a precisão da resposta.

**Sobre a Indexação Vetorial (IVFFlat) e Pesquisa Híbrida (Hybrid Search):**
- De acordo com as boas práticas listadas na **Apresentação 5, Slide 18**, aplicámos um índice **IVF (Inverted File Index)** na nossa base de dados. Em vez de forçar *Full Table Scans* lentos e exaustivos, o IVF agrupa vetores, reduzindo significativamente o custo computacional da recuperação.
- Adicionalmente, implementámos *Hybrid Search* (pesquisa semântica por cosseno + Full-Text Search via GIN index). Unimos os dois através do algoritmo RRF (Reciprocal Rank Fusion). Isto resolve o clássico problema do RAG onde a pesquisa semântica perde correspondências exatas de palavras-chave (ex: nomes próprios ou IDs).

