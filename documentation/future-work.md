# Trabalho Futuro

## Red-Teaming Continuo Automado (Garak)

Embora as nossas regras determinísticas em Colang tenham estabelecido com sucesso a segurança de base, um próximo passo crítico para a implementação em produção é a integração de um varredor de vulnerabilidades automatizado como o Garak. 

Ao encaminhar as sondagens adversariais do Garak (por exemplo, *jailbreaks* do tipo DAN ou injeções de *prompts*) através do nosso *wrapper* FastAPI, podemos testar matematicamente os nossos NeMo Guardrails sob esforço, gerando relatórios empíricos sobre a nossa postura de defesa e descobrindo vulnerabilidades em casos limite (*edge cases*) antes que agentes maliciosos o façam.

## Sistema de Prevenção de Intrusões ao Nível da API (Fail2Ban)

Atualmente, o nosso sistema intercepta com segurança intenções maliciosas usando NeMo guardrails, mas continua a gastar ciclos de CPU a calcular *embeddings* para Bad Actors. Para mitigar ataques de Negação de Carteira (*Denial of Wallet* - DoW) e de exaustão de computação, planeamos implementar um Sistema de Prevenção de Intrusões ao nível da API. 

Ao rastrear as métricas de recusa dos NeMo Guardrails através da nossa API Gateway, podemos aplicar uma política de "Três Faltas" (*Three-Strikes*), colocando automaticamente endereços IP maliciosos numa lista negra ao nível da *firewall* e rejeitando tráfego hostil antes que este atinja a aplicação em Python.

## Otimização Programática de Prompts

Devido a limitações de VRAM local, os nossos *prompts* de sistema e esquemas de avaliação foram concebidos manualmente. Num ambiente em nuvem escalado, implementaríamos o GenAI Prompt Optimizer do MLflow. Ao introduzir o nosso *Golden Set* estabelecido nesta *pipeline*, poderíamos utilizar um Meta-LLM para iterativamente alterar e avaliar os nossos *prompts* de RAG, maximizando matematicamente as nossas métricas de Similaridade de Jaccard e Fidelidade sem intervenção humana.