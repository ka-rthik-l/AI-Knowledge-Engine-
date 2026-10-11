"""Command-line entrypoint for the AI Knowledge Engine."""

import os
import sys
from config import THRESHOLD
from core.chunker import chunk_text
from core.generator import generate_answer
from core.semantic_search import SemanticSearch
from ingestion.pdf_loader import load_pdf


def build_or_load_index(engine: SemanticSearch, data_folder: str = "data") -> None:
    """Loads existing vector index or builds a new one from PDF files.

    Args:
        engine: SemanticSearch engine instance.
        data_folder: Folder containing PDF files to index.
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


def main() -> None:
    """Main execution loop for user queries."""
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

    print(f"\nTop {len(results)} matching passages:\n")
    for i, (entry, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"Source: {entry.get('source', 'Unknown')}")
        print(entry["text"])
        print("-" * 50)

    context = "\n".join([entry["text"] for entry, score in results])

    print("\nGenerating answer...")
    try:
        answer = generate_answer(query, context)
        print("\nAnswer:\n")
        print(answer)
    except Exception as exc:
        print(f"\nCould not generate answer: {exc}")


if __name__ == "__main__":
    main()