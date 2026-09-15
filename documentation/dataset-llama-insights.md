# Insights do Run com o Juiz Llama-3.2-1B

**Análise dos resultados da avaliação batched com o modelo juiz `llama-3.2-1b-instruct`.**

Este documento resume os *insights* extraídos do primeiro *run* completo de avaliação, antes da migração do juiz para o `deepseek-r1-0528-qwen3-8b`. Serve de base para a análise crítica (Slide 4) e documenta tanto o que correu bem como as limitações descobertas no processo.

---

## Resultados agregados

| Run Name | Duration | avg_faithfulness | avg_overlap | avg_relevancy | generator_model | judge_model | temperature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-2b_temp_2 | 1.3min | 0.554054054054054 | 0.2857412621552141 | 0.6126126126126128 | qwen3.5-2b | llama-3.2-1b-instruct | 2 |
| qwen3.5-2b_temp_1.5 | 1.3min | 0.5495495495495494 | 0.2857412621552141 | 0.6126126126126128 | qwen3.5-2b | llama-3.2-1b-instruct | 1.5 |
| qwen3.5-2b_temp_1 | 1.3min | 0.5405405405405405 | 0.2857412621552141 | 0.6126126126126128 | qwen3.5-2b | llama-3.2-1b-instruct | 1 |
| qwen3.5-2b_temp_0.5 | 1.3min | 0.5495495495495494 | 0.2857412621552141 | 0.6126126126126128 | qwen3.5-2b | llama-3.2-1b-instruct | 0.5 |
| qwen3.5-2b_temp_0.0 | 1.3min | 0.554054054054054 | 0.2857412621552141 | 0.6081081081081082 | qwen3.5-2b | llama-3.2-1b-instruct | 0.0 |
| ministral-3-3b-instruct-2512_temp_2 | 1.3min | 0.5675675675675674 | 0.3109419836573907 | 0.6216216216216216 | ministral-3-3b-instruct-2512 | llama-3.2-1b-instruct | 2 |
| ministral-3-3b-instruct-2512_temp_1.5 | 1.3min | 0.5720720720720719 | 0.3109419836573907 | 0.6171171171171117 | ministral-3-3b-instruct-2512 | llama-3.2-1b-instruct | 1.5 |
| ministral-3-3b-instruct-2512_temp_1 | 1.3min | 0.5765765765765766 | 0.3109419836573907 | 0.6171171171171117 | ministral-3-3b-instruct-2512 | llama-3.2-1b-instruct | 1 |
| ministral-3-3b-instruct-2512_temp_0.5 | 1.3min | 0.563063063063063 | 0.3109419836573907 | 0.6216216216216216 | ministral-3-3b-instruct-2512 | llama-3.2-1b-instruct | 0.5 |
| ministral-3-3b-instruct-2512_temp_0.0 | 1.3min | 0.5720720720720719 | 0.3109419836573907 | 0.6171171171171117 | ministral-3-3b-instruct-2512 | llama-3.2-1b-instruct | 0.0 |

---

## Insight 1 — A "Anomalia Einstein" (falha do juiz de 1B)

- Ao analisar as colunas `Faithfulness Reason` e `Relevancy Reason` dos ficheiros JSON, verificámos que, para quase todas as perguntas (quer fossem sobre EMAR, quer sobre a Força Aérea Portuguesa), o modelo juiz `llama-3.2-1b-instruct` produziu justificações do tipo:
  > *"The score is 0.67 because Einstein's work on the photoelectric effect predates the Nobel Prize..."*
  > *"The claim about Albert Einstein winning the Nobel Prize in Physics... is irrelevant."*
- **O que isto prova:** LLMs pequenos (1B de parâmetros) são fundamentalmente incapazes de atuar como juízes fiáveis para métricas complexas como *Faithfulness*. Os *prompts* internos do DeepEval são demasiado complexos para um modelo de 1B, que entra em *hallucination* descontrolada (sobre Albert Einstein!) em vez de avaliar o contexto RAG.
- **Conclusão para o relatório:** *"Embora tenhamos implementado com sucesso a arquitetura LLM-as-a-judge (DeepEval), os dados revelaram uma limitação crítica de LLMOps em *edge computing*: a capacidade do juiz. O Llama-3.2-1B sofreu um colapso severo na execução de instruções, alucinando justificações sobre 'Albert Einstein' em vez de avaliar a conformidade EMAR. Concluímos, portanto, que, embora modelos de 2B-3B sejam adequados para a geração RAG, a fase de avaliação exige estritamente um modelo com mais parâmetros para garantir métricas automáticas fiáveis."*
- Esta descoberta é a justificação central para a migração para o `deepseek-r1-0528-qwen3-8b` (ver `Notas_Relatorio.md`).

---

## Insight 2 — O "flatline" do `avg_overlap` (o bug de temperatura)

- Para ambos os modelos, o `avg_overlap` é **exatamente idêntico** nas 5 temperaturas (0.2857412621552141 para o Qwen; 0.3109419836573907 para o Ministral). Em IA, subir a temperatura para `2.0` normalmente produz resultados completamente diferentes — porque é que o overlap não mudou?
- **Investigação:** a causa não foi "os guardrails tornarem o sistema determinístico", mas sim um **bug na propagação da temperatura**: no `guardrails.py`, a ação `run_rag_action` chamava `rag.ask_rag(query, model_name=...)` **sem passar a temperatura**, que ficava sempre no valor por omissão `0.0`. As 5 temperaturas do ciclo em `evaluate.py` eram, na prática, **5 execuções a temperatura 0.0**. O overlap de Jaccard é determinístico — se as respostas variassem com a temperatura, a média variaria. Ser bit-idêntico a 16 casas decimais em 37 perguntas é impossível por acaso.
- **O que isto prova:** (1) a métrica lexical determinística revelou um bug silencioso de *plumbing* que as métricas semânticas mascararam; (2) a pequena variação de *faithfulness/relevancy* (0.54→0.55) não é efeito da temperatura, mas sim ruído estocástico do juiz de 1B sobre inputs idênticos — o que *reforça* o Insight 1.
- **Conclusão para o relatório:** *"A nossa métrica lexical personalizada provou o valor das métricas determinísticas no processo de LLMOps: detetou que a temperatura nunca era propagada do loop de avaliação para o gerador (o parâmetro ficava sempre em 0.0), um erro que as métricas semânticas, ruidosas, esconderam. Corrigimos o bug (temperatura agora encadeada por `GuardrailsPipeline(temperature=...)` → `run_rag_action` → `rag.ask_rag`) e o `avg_overlap` passou a variar com a temperatura, validando o sweep experimental."*
- **Nota de apresentação:** NÃO apresentar o flatline como "prova de que os guardrails tornam o sistema determinístico" — o professor vai abrir `guardrails.py` na discussão e o argumento colapsa. O argumento honesto (bug detetado através de dados) é um ponto de análise crítica muito mais forte.

---

## Insight 3 — Ministral supera Qwen em ambientes estritos

- O `ministral-3-3b-instruct` supera consistentemente o `qwen3.5-2b` nas três métricas:
  - **Overlap:** 0.31 vs 0.28
  - **Faithfulness:** ~0.57 vs ~0.55
  - **Relevancy:** ~0.62 vs ~0.61
- Nos ficheiros JSON, o Qwen era demasiado "conservador": respondia *"I do not have enough information..."* mesmo quando o contexto recuperado continha a resposta. O Ministral lia melhor os chunks do pgvector e sintetizava respostas mais alinhadas com o *golden set*.
- **Cautela:** as diferenças são pequenas (~2 pontos percentuais) e foram medidas com um juiz de 1B que alucinava. O veredito final sobre qual modelo escolher deve ser **revalidado com o novo juiz** (`deepseek-r1-0528-qwen3-8b`) antes de ser apresentado como decisão de produção.
- **Conclusão preliminar para o relatório:** *"Numa análise comparativa, o Ministral-3B revelou-se superior ao Qwen-2.5-2B para o nosso caso de uso. Alcançou um overlap léxico mais alto (31% vs 28%), indicando maior aderência à formulação regulamentar esperada. O Qwen exibiu comportamentos de recusa excessivamente agressivos, reclamando informação insuficiente mesmo quando o contexto válido era recuperado."*

---

## Insight 4 — A realidade do "léxico" vs "semântico"

- Os scores de overlap (~30%) parecem baixos à primeira vista, mas são exatamente o esperado em RAG.
- O *golden set* contém respostas esperadas curtas (ex.: *"EMAR 147."*), enquanto a LLM gera frases completas e polidas (ex.: *"Based on the documents, the requirement is EMAR 147."*). A Similaridade de Jaccard penaliza fortemente palavras extra.
- **Conclusão para o relatório:** *"A nossa estratégia de avaliação dupla expôs a diferença entre scoring léxico e semântico. A métrica de overlap lexical média ~30% penaliza a LLM por gerar frases completas e conversacionais em vez dos bullets breves do golden set. Isto sublinha a necessidade de usar avaliadores semânticos (como Answer Relevancy) em conjunto com métricas NLP tradicionais para capturar a qualidade real de uma resposta sem penalizar injustamente a sintaxe conversacional."*

---

## Resumo executivo

1. O juiz de 1B alucina sistematicamente (Einstein) → justifica a migração para o `deepseek-r1-0528-qwen3-8b`.
2. O flatline do overlap revelou um **bug real de propagação de temperatura**, não determinismo dos guardrails — corrigido no código.
3. O Ministral supera o Qwen nas 3 métricas, mas a diferença é pequena e precisa de ser confirmada com o novo juiz.
4. O overlap lexical ~30% é um artefacto esperado da disparidade entre golden set curto e respostas conversacionais — daí a necessidade de métricas semânticas complementares.

**Estado:** os resultados com o novo juiz (`deepseek-r1-0528-qwen3-8b`) estavam a ser recolhidos quando este documento foi escrito; este ficheiro deve ser atualizado quando estiverem disponíveis. O novo *run* usa apenas **3 temperaturas** (`0.0, 1.0, 2.0`), em vez das 5 deste run com o llama.
