# nlpg-6ed-group-4

Correr `uv sync`
    - Faz tudo, cria o .venv e instala os pacotes necessários

Correr `docker compose up -d`
	- Levanta um docker de Postgres SQL com o plugin VectorPG para se comportar como uma base de dados vectorial

Correr o MLFlow (num terminal á parte):
    `uvx mlflow server --host 127.0.0.1 --port 5000`
Garantir que o LM_Studio está a correr