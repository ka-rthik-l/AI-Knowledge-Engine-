from core.semantic_search import SemanticSearch
from core.chunker import chunk_text
from ingestion.pdf_loader import load_pdf
from core.generator import generate_answer
import os

engine = SemanticSearch()

try:
    engine.load()
    print("Loaded memory.")
except:
    print("Building new memory...")

    data_folder = "data"

    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            path = os.path.join(data_folder, file)
            print("Indexing:", file)

            text = load_pdf(path)
            chunks = chunk_text(text)

            engine.add(chunks, source=file)

    engine.save() 
    print("Memory saved.")

query = input("\nAsk: ")
results = engine.search(query)



for i, (entry, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.3f}")
    print(f"Source: {entry['source']}")
    print(entry["text"])
    print("-" * 50)

context = "\n".join([entry["text"] for entry, score in results])

answer = generate_answer(query, context)

print("\nAnswer:\n")
print(answer)