# Natural Language Processing (NLP) with Generative Artificial Intelligence (Gen AI)

**GRUPO 4 (nlpg-6ed-group-4):** Vítor Antunes | José Galão | Raquel Rocha

---

## Slide 1: Introdução e Contexto do Problema
* **Dataset Escolhido:** Breve descrição dos dados.
* **Objetivo:** Apresentar dois LLMs de código aberto selecionados para comparação.

---

## Slide 2: Arquitetura do Sistema (Diagramas C4)
* **Nível 1 (System Context):** Mostrar a interação do utilizador com o sistema RAG e as fontes de dados externas.
* **Nível 2 (Container):** Identificar os componentes principais: Base de Dados Vetorial (ex: Chroma, Qdrant ou Milvus), Orquestrador Python e Modelos.
* **Nível 4 (Code Diagram):** Incluir o diagrama de código gerado automaticamente a partir do nosso repositório GitHub.

---

## Slide 3: Implementação Técnica (Building Blocks e RAG)
* **Pipeline de Dados:** Explicar a estratégia de chunking (ex: recursivo ou semântico) e o modelo de embeddings utilizado para povoar a base vetorial.
* **Orquestração:** Descrever como o sistema obtém o conhecimento externo para alimentar o prompt do LLM.
* **Redes Neurais (Opcional):** Referência rápida à compreensão teórica (ex: papel das funções de ativação ou atenção) que sustenta os modelos escolhidos.

---

## Slide 4: Avaliação Comparativa - Análise Crítica
* **Métricas de Performance:** Apresentar resultados quantitativos utilizando métricas como ROUGE, BLEU ou Perplexity.
* **Comparação Qualitativa:** Pesagem de prós e contras entre os dois modelos (velocidade vs. precisão, latência, alucinações).
* **Uso de Ferramentas:** MLFlow para monitorização ou LLM Autorater para avaliação.

---

## Slide 5: Segurança e Conclusão
* **Guardrails Implementados:** Demonstrar os mecanismos de segurança (ex: Nvidia NeMo ou Garak) para evitar toxicidade, alucinações ou fuga de dados, etc.
* **Decisão Final:** Recomendação fundamentada do melhor modelo para a organização.
* **Links:** URL do repositório GitHub com o código e histórico de commits.
