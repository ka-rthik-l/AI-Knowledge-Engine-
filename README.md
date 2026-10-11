# AI Knowledge Engine

A local, lightweight Retrieval-Augmented Generation (RAG) and semantic document retrieval system built using dense sentence embeddings, cosine similarity search, and Ollama.

## Overview

AI Knowledge Engine ingests local PDF documents, splits them into overlapping word chunks, embeds them into high-dimensional vectors, and persists the index to disk. When queried, it retrieves the most semantically relevant text passages using cosine similarity and feeds them as grounded context to a local LLM to generate precise answers.

## Architecture

```
PDFs (data/)
     │
     ▼
[pdf_loader.py] ───► Extract text
     │
     ▼
[chunker.py]    ───► Overlapping word chunks
     │
     ▼
[embeddings.py] ───► SentenceTransformer ("all-MiniLM-L6-v2")
     │
     ▼
[vector_store.py]──► Save numpy vectors & metadata to storage/
     │
     ▼
[semantic_search.py] ◄── User Query
     │
     ▼ (Cosine similarity ranking & threshold filtering)
Top-K Context Passages
     │
     ▼
[generator.py]  ───► Ollama ("phi3")
     │
     ▼
Answer to User
```

## Features

- **Robust PDF Ingestion**: Page-by-page text extraction with validation for corrupted or zero-byte files.
- **Configurable Chunking**: Sliding-window chunking with customizable overlap to preserve sentence boundaries.
- **Dense Vector Search**: Powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and vectorized cosine similarity.
- **Fast Local Persistence**: Zero external database dependencies; stores dense vectors in `.npy` format and metadata in UTF-8 `.json`.
- **Threshold & Top-K Filtering**: Configurable similarity cutoffs to prevent hallucination from low-relevance passages.
- **Local LLM Grounding**: Prompts Ollama with strictly constrained context instructions.
- **Comprehensive Test Suite**: Unit tests covering boundary conditions, edge cases, error resilience, and mocking.

## Project Structure

```
AI-Knowledge-Engine-/
├── config.py             # Hyperparameters and environment configuration
├── main.py               # CLI entrypoint and interactive search loop
├── requirements.txt      # Project Python dependencies
├── core/
│   ├── chunker.py         # Word-level text chunking with overlap
│   ├── embeddings.py      # Lazy-loaded sentence embeddings
│   ├── generator.py       # Ollama LLM response synthesis
│   ├── semantic_search.py # Cosine similarity ranking and search engine
│   └── vector_store.py    # NumPy and JSON index persistence
├── ingestion/
│   └── pdf_loader.py      # PyPDF text extraction and error handling
├── data/                  # Source PDF documents for indexing
├── storage/               # Saved vector index and metadata
└── tests/                 # Unit test suite
    ├── test_chunker.py
    ├── test_generator.py
    ├── test_pdf_loader.py
    ├── test_semantic_search.py
    └── test_vector_store.py
```

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** (for answer generation)

## Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/ka-rthik-l/AI-Knowledge-Engine-.git
cd AI-Knowledge-Engine-

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama Model

Ensure the Ollama application or daemon is running:

```bash
ollama pull phi3
```

### 4. Add Documents & Run

Place your `.pdf` documents into the `data/` directory, then start the engine:

```bash
python main.py
```

On first run, the system will automatically index all documents in `data/` and persist the embeddings to `storage/`. On subsequent runs, it reloads the saved index immediately.

## Configuration

All hyperparameters in `config.py` can be customized via environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `120` | Number of words per text chunk |
| `CHUNK_OVERLAP` | `30` | Number of overlapping words between consecutive chunks |
| `TOP_K` | `5` | Maximum number of passages retrieved for LLM context |
| `THRESHOLD` | `0.3` | Minimum cosine similarity score (0.0 to 1.0) |
| `STORE_PREFIX` | `storage/store` | Path prefix for index files |
| `LLM_MODEL` | `phi3` | Ollama model tag used for generation |

Example override on Windows (PowerShell):
```powershell
$env:TOP_K="3"; $env:THRESHOLD="0.4"; python main.py
```

## Running Tests

Run the complete unit test suite using Python's built-in `unittest`:

```bash
python -m unittest discover tests
```

## License

MIT License.
