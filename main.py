"""Command-line entrypoint for the AI Knowledge Engine.

Orchestrates document indexing, vector similarity retrieval,
and grounded LLM answer generation in an interactive loop.
"""

# ==============================================================================
# 1. IMPORTS & SYSTEM CONFIGURATION
# ==============================================================================

import os
import sys
from typing import List, Tuple, Dict, Any
from config import THRESHOLD
from core.chunker import chunk_text
from core.generator import generate_answer
from core.semantic_search import SemanticSearch
from ingestion.pdf_loader import load_pdf


# ==============================================================================
# 2. DOCUMENT INGESTION & INDEX LIFECYCLE
# ==============================================================================

def build_or_load_index(engine: SemanticSearch, data_folder: str = "data") -> None:
    """Loads an existing vector index or builds a new one from PDF files.

    Args:
        engine: SemanticSearch engine instance.
        data_folder: Folder containing PDF files to ingest and index.
    """
    try:
        engine.load()
        print("Loaded memory from disk.")
    except (FileNotFoundError, OSError):
        print("Existing index not found. Building new memory...")
        os.makedirs(data_folder, exist_ok=True)
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"No PDF documents found in '{data_folder}'.")
            return

        for file in pdf_files:
            path = os.path.join(data_folder, file)
            print(f"Indexing: {file}")
            text = load_pdf(path)
            chunks = chunk_text(text)

            if chunks:
                engine.add(chunks, source=file)
            else:
                print(f"  Warning: No text extracted from '{file}'.")

        engine.save()
        print("Memory saved to disk.")


# ==============================================================================
# 3. CONTEXT ASSEMBLY & QUERY PIPELINE
# ==============================================================================

def display_search_results(results: List[Tuple[Dict[str, Any], float]]) -> None:
    """Prints ranked matching passages with sources and similarity scores.

    Args:
        results: List of (entry_dict, score) tuples.
    """
    print(f"\nTop {len(results)} matching passages:\n")
    for i, (entry, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"Source: {entry.get('source', 'Unknown')}")
        print(entry["text"])
        print("-" * 50)


# ==============================================================================
# 4. INTERACTIVE CLI RUNTIME LOOP
# ==============================================================================

def main() -> None:
    """Main execution loop for user queries and generative retrieval."""
    engine = SemanticSearch()
    build_or_load_index(engine)

    try:
        query = input("\nAsk: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    if not query:
        print("Query cannot be empty.")
        return

    results = engine.search(query)

    if not results:
        print(f"\nNo results found with similarity score >= {THRESHOLD:.2f}.")
        return

    display_search_results(results)

    context = "\n".join([entry["text"] for entry, score in results])

    print("\nGenerating answer...")
    try:
        answer = generate_answer(query, context)
        print("\nAnswer:\n")
        print(answer)
    except Exception as exc:
        print(f"\nCould not generate answer: {exc}")


# ==============================================================================
# 5. SCRIPT EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    main()