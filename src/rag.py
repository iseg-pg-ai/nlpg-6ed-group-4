import os

from dotenv import load_dotenv

import retriever
from helpers.lm_studio_utils import LMStudioModel, LMStudioModelEmbedder

load_dotenv()  # Load environment variables


def build_prompt(context_chunks: list[str]) -> str:
    """Constructs the system prompt strictly enforcing the context."""
    context_str = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are an expert AI assistant. Your task is to answer the user's question "
        "using ONLY the information provided in the context below.\n"
        "If the answer cannot be found in the context, reply exactly with: "
        "'I do not have enough information to answer this based on the retrieved documents.'\n\n"
    )
    # This prompt acts as a "soft guardrail" against hallucinations
    return system_prompt + f"CONTEXT:\n{context_str}"


def ask_rag(query: str, model_name: str, temperature: float = 0.0) -> dict:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    Returns a dictionary containing the answer and the retrieved context.
    """
    print(f"\n[RAG] Retrieving context for query: '{query}'...")

    contexts = retriever.retrieve_context(query)  # Retrieve the context using the PGVector setup

    if not contexts:  # Check if we got any context (basic guardrail)
        return {
            "answer": "No relevant documents were found in the database.",
            "context": [],
            "model": model_name,
        }

    system_message = build_prompt(contexts)

    llm = LMStudioModel(model_name)
    print(f"[RAG] Sending prompt to {model_name}...")
    try:
        response = llm.generate(  # Query the local LLM
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            temperature=temperature,  # 0.0 is best for factual RAG to prevent creativity/hallucination
            max_tokens=500,
        )

        message = response.choices[0].message

        # Convert the strict Pydantic model into a normal dictionary to bypass strict typing
        message_dict = message.model_dump()

        answer = message_dict.get("content") or ""  # Safely extract standard content
        reasoning = message_dict.get("reasoning_content")  # check for reasoning_content (Qwen-distils models)
        if reasoning:
            print("[SYSTEM] Reasoning tokens detected!")  # Append the reasoning answer so you can see both
            answer = f"<thought_process>\n{reasoning}\n</thought_process>\n\n{answer}"

        if not answer.strip():  # 3. Fallback: Did it still return nothing?
            print("[WARNING] The model returned an empty string.")
            if message_dict.get("tool_calls"):
                answer = "[ERROR: The model attempted a tool call instead of answering.]"
            else:
                answer = "[ERROR: Empty response. If using Qwen, ensure LMStudio is set to the 'ChatML' prompt format!]"

    except Exception as e:
        print(f"[ERROR] Failed to connect to LLM: {e}")
        print("Make sure llama.cpp or LMStudio server is running!")
        answer = "Error generating response."

    # 4. Return BOTH answer and context, returning the context is crucial to evaluate
    return {"answer": answer, "context": contexts, "model": model_name}


def test_rag(test_query):
    # Ensure your llama.cpp/LMStudio server is running before executing this!
    model_a = os.getenv("MODEL_A", "mistralai/ministral-3-3b")
    model_b = os.getenv("MODEL_B", "qwen_qwen3.5-2b")
    embed_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")

    LMStudioModel.unload_all()

    LMStudioModelEmbedder(embed_model).load()  # Load CPU embedder
    LMStudioModel(model_a).load()  # Load GPU LLM
    print(f"--- Testing RAG Pipeline with {model_a} ---")
    result = ask_rag(test_query, model_a)
    print(f"\n=== FINAL ANSWER WITH {model_a} ===")
    print(result["answer"])
    LMStudioModel(model_a).unload()

    LMStudioModel.unload_all()
    LMStudioModelEmbedder(embed_model).load()  # Reload CPU embedder
    LMStudioModel(model_b).load()  # Load GPU LLM
    print(f"--- Testing RAG Pipeline with {model_b} ---")
    result = ask_rag(test_query, model_b)
    print(f"\n=== FINAL ANSWER WITH {model_b} ===")
    print(result["answer"])
    LMStudioModel(model_b).unload()


if __name__ == "__main__":
    test_rag("What is the main topic of the text?")
