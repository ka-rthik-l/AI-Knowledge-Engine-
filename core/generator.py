"""Answer generation using local LLM models via Ollama."""

import inspect
import logging
from typing import Optional
from config import LLM_MODEL

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    context: str,
    model: str = LLM_MODEL,
) -> str:
    """Generates an answer grounded strictly in the provided retrieval context.

    Args:
        query: The user question.
        context: Concatenated text passages retrieved from semantic search.
        model: Ollama model identifier to use for inference.

    Returns:
        The generated answer string.

    Raises:
        ImportError: If the ollama package is not installed.
        RuntimeError: If communication with the Ollama service fails.
    """
    if not context or not context.strip():
        return "No relevant context found in documents to answer the question."

    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' library is required for generative answering. "
            "Please install it with 'pip install ollama' and ensure Ollama is running."
        ) from exc

    prompt = inspect.cleandoc(f"""
        Answer the question using ONLY the context below.
        If the answer cannot be determined from the context, state that clearly.

        Context:
        {context}

        Question:
        {query}
    """)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as exc:
        logger.error("Ollama inference failed with model '%s': %s", model, exc)
        raise RuntimeError(
            f"Failed to generate answer with Ollama model '{model}': {exc}"
        ) from exc
