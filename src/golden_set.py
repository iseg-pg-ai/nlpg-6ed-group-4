"""
This file contains the "Golden Set" for evaluating our RAG pipeline.
"""

EVALUATION_DATA = [
    {
        "query": "What is the main topic of the text?",
        "expected_output": "The main topic revolves around a detective investigation, specifically analyzing handwriting and sewing marks to identify a typist.",
    },
    {
        "query": "Who is the main character mentioned?",
        "expected_output": "Sherlock Holmes is the main character conducting the investigation.",
    },
    # Esquema Tipo:
    # {
    #     "query": "...",
    #     "expected_output": "..."
    # }
]
