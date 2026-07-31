---
puppeteer:
  pdf:
    printBackground: true
    margin:
      top: 0
      bottom: 0
      left: 0
      right: 0
---

# Ciclo do Pipeline: INGEST → RETRIEVE → GUARDRAILS → EVALUATE

**Guia de exploração do código**

Este documento descreve o funcionamento do projeto do início ao fim: o **ciclo do pipeline RAF**. 
Cada etapa (ou *stage*) tem um número (1 a 4) e cada sub-etapa tem uma letra (ex.: `3c`).

Quando virem estas referências no código (como comentários `# STAGE 3c-i`) ou neste documento, elas apontam sempre para o ficheiro onde aquele passo está implementado.

> **Como usar este documento:** leiam primeiro o resumo visual, depois as etapas 1 → 4 pela ordem apresentada. Cada etapa indica o ficheiro a abrir, o que faz, e como se liga à etapa seguinte.

---

## 1. Visão geral — o que é este projeto

Este projeto é um **sistema RAG** (*Retrieval-Augmented Generation*): uma LLM (*Large Language Model*, modelo de linguagem) que responde a perguntas sobre os regulamentos **EMAR** (normas militares de aeronavegabilidade), mas **apenas** com base em documentos oficiais que estão na pasta `data/`, nunca inventa respostas.

Para isso, o sistema executa um **ciclo de 4 etapas**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        O CICLO DO PIPELINE                               |
│                                                                          │
│  1. INGEST ───► 2. RETRIEVE ───► 3. GUARDRAILS/RAG ───► 4. EVALUATE      │
│  (prepara os   (procura a     (verifica segurança   (mede a qualidade)   │
│   documentos)   informação)     e gera a resposta)                       │
│                                                                          │
│   ingest.py     retriever.py    guardrails.py          evaluate.py       │
│                                 └── rag.py                               │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Etapas 1 e 2** preparam e recuperam a informação (a "base de conhecimento").
- **Etapa 3** filtra perguntas proibidas e gera a resposta.
- **Etapa 4** avalia se as respostas são boas, registando os resultados no MLflow.

O ponto de entrada (`src/__main__.py`) executa, por agora, apenas **1 → 4**, porque as etapas 2 e 3 são chamadas pela etapa 4.

---

## 2. Etapa 1 — INGEST (Ingestão de documentos)

**Ficheiro:** `src/ingest.py`

**Objetivo:** transformar os PDFs/TXTs da pasta `/data` numa base de dados *vetorial* á qual o sistema consegue " fazer perguntas". Sem esta etapa, as restantes não têm nada para procurar.

### Porquê "vetorial"?

Uma LLM não lê texto como nós. Para ela, um documento é uma lista de números (o **vetor de embedding**). Textos com significados parecidos ficam com números (Vetores são conjuntos de números) parecidos. Esta etapa calcula esses vetores e guarda-os na base de dados **pgvector** (PostgreSQL com suporte vetorial), numa tabela chamada `document_chunks`.

### Sub-etapas

| Código | Nome | Descrição |
|:------:|------|-----------|
| `1a` | **EXTRACT** | Lê o PDF/TXT e extrai o texto bruto, página a página. |
| `1b` | **CHUNK** | Parte o texto em pedaços pequenos (~500 caracteres) alinhados por frases. |
| `1c` | **EMBED** | Converte cada pedaço/chunk num vetor numérico, num só pedido em lote ao LM Studio. |
| `1d` | **PERSIST** | Grava cada `(ficheiro, pedaço, vetor)` na tabela `document_chunks`. |

> **Nota importante:** os pedaços (`chunks`) têm **500 caracteres** (~130 *tokens*), não tokens. O corte respeita frases completas, ou seja nunca parte uma palavra a meio. Para blocos de PDF enormes sem pontuação existe um mecanismo de segurança que corta por caracteres.

### Fluxo (com ficheiro)

1. **`1a` — EXTRACT** → `extract_text_from_pdf()` em `src/ingest.py`
2. **`1b` — CHUNK** → `chunk_text()` em `src/ingest.py`
3. **`1c` — EMBED** → `LMStudioModelEmbedder.embed_batch()` em
   `src/helpers/lm_studio_utils.py`
4. **`1d` — PERSIST** → `db.insert_chunks_batch()` em
   `src/helpers/db_utils.py`

> As instruções de configuração (modelo de embedding, tamanho do pedaço/chunk, sobreposição) estão em `.env` e são lidas pela classe `IngestConfig` em `src/ingest.py`.

**Ligação à etapa seguinte:** quando um utilizador faz uma pergunta, o sistema vai procurar nesta tabela os pedaços/chunks mais relevantes.

---

## 3. Etapa 2 — RETRIEVE (Recuperação de contexto)

**Ficheiro:** `src/retriever.py`

**Objetivo:** dada uma pergunta, devolver os **pedaços/chunks de documento mais relevantes** para a responder. É a "pesquisa" do sistema.

### Sub-etapas

| Código | Nome | Descrição |
|:------:|------|-----------|
| `2a` | **QUERY EMBEDDING** | Converte a pergunta num vetor, usando o mesmo modelo de embedding da Etapa 1. |
| `2b` | **HYBRID SEARCH** | Pesquisa de duas formas em paralelo e combina os resultados (RRF). |
| `2c` | **RE-RANK** (opcional) | Uma LLM volta a ordenar os resultados para pôr os melhores primeiro. |

### O que é "pesquisa híbrida" (2b)?

São duas pesquisas independentes, cujos resultados são combinados:

1. **Pesquisa vetorial** — encontra pedaços/chunks *semanticamente* parecidos com a pergunta (mesmo que usem palavras diferentes).
2. **Pesquisa de texto integral (FTS)** — encontra pedaços/chunks que contêm as palavras exatas da pergunta.

A combinação usa **RRF** (*Reciprocal Rank Fusion*): dá pontos por posição em cada lista e junta as duas. Um pedaço/chunk com boa performance em ambas sobe na classificação.

### Fluxo (com ficheiro)

1. **`2a` — QUERY EMBEDDING** → `LMStudioModelEmbedder.embed()` em
   `src/helpers/lm_studio_utils.py`
2. **`2b` — HYBRID SEARCH** → `db.search_chunks_hybrid()` em
   `src/helpers/db_utils.py` (que chama `search_chunks_vector()` +
   `search_chunks_fts()`)
3. **`2c` — RE-RANK** → `rerank()` em `src/reranker.py`

A função pública que orquestra tudo é `retrieve_context()` em `src/retriever.py`. Os resultados são **guardados em cache** (memória) por pergunta, para não repetir pesquisas iguais.

**Ligação à etapa seguinte:** os pedaços/chunks recuperados são o "contexto" que vai ser entregue à LLM para ela responder.

---

## 4. Etapa 3 — GUARDRAILS / RAG (Proteção e geração)

**Ficheiro principal:** `src/guardrails.py`
**Ficheiro de geração:** `src/rag.py`

**Objetivo:** duas coisas em sequência:

1. **Proteger** — impedir que o sistema responda a perguntas proibidas (política, ilegal, tóxica, classificada, tentativas de "hackar" o prompt).
2. **Gerar** — para perguntas válidas, produzir a resposta com base no contexto.

### Sub-etapas (guardrails)

| Código | Nome | Descrição |
|:------:|------|-----------|
| `3a` | **INTENT MATCHING** | O NeMo compara a pergunta com regras Colang (em `nemo_config/`) para decidir se é proibida. |
| `3b` | **RESPONSE** | Se for proibida → resposta de recusa direta. Se for válida → chama o RAG. |
| `3c` | **RAG EXECUTION** | Executa a ação `run_rag_action`, que delega no `rag.py`. |

> Os guardrails são feitos com **NeMo Guardrails**. As regras estão em `nemo_config/rails.co` (frases que disparam recusas) e `nemo_config/config.yml` (modelos e deteção de PII (Personally Identifiable Information) ). A configuração é copiada para uma pasta temporária com as variáveis substituídas e o ambiente global não é alterado.

### Sub-etapas do RAG (3c, dentro do ficheiro `rag.py`)

Quando a pergunta é válida, o RAG executa três passos internos:

| Código | Nome | Descrição |
|:------:|------|-----------|
| `3c-i` | **RETRIEVE** | Chama a Etapa 2 (`retriever.retrieve_context()`) para obter o contexto. |
| `3c-ii` | **GROUND** | Processo de "Ancoragem" do modelo: insere o contexto num prompt de sistema que obriga a responder só com os documentos. |
| `3c-iii` | **GENERATE** | Envia o prompt ao LM Studio e recebe a resposta da LLM. |

O *grounding* (`3c-ii`) é o coração do RAG: o sistema é instruído a responder **apenas** com o contexto fornecido e, se a resposta não existir nos documentos, a dizer "não tenho informação suficiente".

### Fluxo (com ficheiro)

1. **`3a` — INTENT MATCHING** → `GuardrailsPipeline.ask()` em
   `src/guardrails.py` (usa `nemo_config/rails.co`)
2. **`3b` — RESPONSE** → `GuardrailsPipeline.ask()` em `src/guardrails.py`
3. **`3c` — RAG EXECUTION** → `run_rag_action()` em `src/guardrails.py`
   - **`3c-i` RETRIEVE** → `retriever.retrieve_context()` em `src/retriever.py`
   - **`3c-ii` GROUND** → `rag.build_prompt()` em `src/rag.py`
   - **`3c-iii` GENERATE** → `LMStudioModel.generate()` em
     `src/helpers/lm_studio_utils.py`

**Ligação à etapa seguinte:** a resposta **e o contexto utilizado** (os pedaços/chunks recuperados) são devolvidos, porque a avaliação precisa de ambos.

---

## 5. Etapa 4 — EVALUATE (Avaliação e registo)

**Ficheiro:** `src/evaluate.py`

**Objetivo:** medir, de forma objetiva, se as respostas do sistema são corretas e fiéis aos documentos. Os resultados são registados no **MLflow** para comparação entre modelos e temperaturas.

### As duas fases

A avaliação funciona em **duas fases**:

- **Fase 1 — Geração:** para cada modelo × temperatura, percorre todas as perguntas do *golden set* e recolhe as respostas (repete todo o ciclo: guardrails → RAG → retriever).
- **Fase 2 — Julgamento:** um modelo "juiz" (imparcial) pontua cada resposta.

### Sub-etapas

| Código | Nome | Descrição |
|:------:|------|-----------|
| `4a` | **CYCLE** | Itera o ciclo completo (Etapa 3) para cada pergunta do *golden set*. |
| `4b` | **SCORE** | O juiz (DeepEval) pontua cada resposta com métricas. |
| `4c` | **AGGREGATE & LOG** | Calcula médias e regista métricas + tabela detalhada no MLflow. |

### O que é o *golden set*?

É a lista oficial de **perguntas com respostas esperadas** usada para testar o sistema. Está em `src/golden_set.py` e é documentada em `documentation/Golden_Set.md`.

Inclui perguntas:
- de recuperação (o sistema deve encontrar a resposta);
- de raciocínio entre vários EMAR;
- em que **não existe** resposta (o sistema deve recusar);
- de segurança (política, jailbreak, etc. — o sistema deve recusar).

### As métricas (em `src/helpers/eval_utils.py` + DeepEval)

| Métrica | O que mede | Como é calculada |
|---------|------------|------------------|
| **Faithfulness** | A resposta é **fiel ao contexto**? (não inventa) | Juiz LLM compara resposta com os pedaços/chunks recuperados. |
| **Answer Relevancy** | A resposta é **relevante** para a pergunta? | Juiz LLM pontua a adequação da resposta. |
| **Lexical Overlap** (Jaccard) | Quantas palavras a resposta partilha com a resposta esperada | Semelhança de conjuntos de palavras (não usa LLM, puramente deterministico). |

> A métrica *Lexical Overlap* é um score **tradicional** (bag-of-words/Jaccard), implementado em `NLPLexicalOverlapMetric` em `src/helpers/eval_utils.py`. As outras duas usam o **LLM como juiz** (`LMStudioJudge`, também em `src/helpers/eval_utils.py`).

### Parâmetros testados

O sistema testa **2 modelos** (Ministral 3B e Qwen 3.5 2B) a **5 temperaturas** (`0.0, 0.5, 1, 1.5, 2`). Temperatura mais baixa = resposta mais determinística; mais alta = mais criativa (mas com mais risco de inventar). Isto permite estudar o trade-off entre fidelidade e criatividade.

### Fluxo (com ficheiro)

1. **`4a` — CYCLE** → `generate_batch_answers()` em `src/evaluate.py`
   (que chama `GuardrailsPipeline.ask()` em `src/guardrails.py`)
2. **`4b` — SCORE** → métricas em `src/helpers/eval_utils.py` + DeepEval
   (juiz `LMStudioJudge` → `LMStudioModel.generate()` em
   `src/helpers/lm_studio_utils.py`)
3. **`4c` — AGGREGATE & LOG** → `mlflow.log_metric()` / `mlflow.log_table()` em
   `src/evaluate.py`

---

## 6. Resumo — mapa rápido do código

| Etapa | Sub-etapas | Ficheiro | Funções principais |
|:-----:|:----------:|----------|--------------------|
| **1. INGEST** | `1a` `1b` `1c` `1d` | `src/ingest.py` | `run_ingestion()`, `chunk_text()`, `extract_text_from_pdf()` |
| **2. RETRIEVE** | `2a` `2b` `2c` | `src/retriever.py` + `src/helpers/db_utils.py` | `retrieve_context()`, `search_chunks_hybrid()`, `rerank()` |
| **3. GUARDRAILS/RAG** | `3a` `3b` `3c` (+`3c-i/ii/iii`) | `src/guardrails.py` + `src/rag.py` | `GuardrailsPipeline.ask()`, `run_rag_action()`, `ask_rag()` |
| **4. EVALUATE** | `4a` `4b` `4c` | `src/evaluate.py` + `src/helpers/eval_utils.py` | `run_evaluation()`, `generate_batch_answers()`, métricas |

**Código comum (usado por todas as etapas):**

| Ficheiro | Papel |
|----------|-------|
| `src/helpers/lm_studio_utils.py` | Gerir modelos no LM Studio (carregar/descarregar, gerar, embeddings). |
| `src/helpers/db_utils.py` | Acesso ao PostgreSQL/pgvector (esquema, inserção, pesquisa). |
| `src/helpers/eval_utils.py` | Juiz LLM para DeepEval + métrica lexical Jaccard. |
| `src/golden_set.py` | O conjunto de perguntas/respostas de avaliação. |
| `nemo_config/` | Regras de guardrails (Colang) e configuração do NeMo. |

---

## 7. Glossário rápido

| Termo | Significado |
|-------|-------------|
| **RAG** | Técnica que combina recuperação de documentos com geração de texto. |
| **Embedding** | Números que representam o "significado" de um texto; textos parecidos → números parecidos. |
| **Chunk** | Pedaço de documento (aqui ~500 caracteres) guardado e pesquisável. |
| **pgvector** | Extensão do PostgreSQL que permite pesquisa vetorial. |
| **RRF** | Método de combinar resultados de várias pesquisas por posição. |
| **Guardrails** | Regras que impedem respostas proibidas ou inseguras. |
| **Golden set** | Conjunto de perguntas com respostas esperadas para avaliar o sistema. |
| **Temperatura** | Controla a criatividade da LLM (0 = determinístico, alto = criativo). |
| **MLflow** | Ferramenta para registar e comparar experiências/resultados. |
| **DeepEval** | Framework de avaliação de LLMs (métricas + "LLM como juiz"). |
