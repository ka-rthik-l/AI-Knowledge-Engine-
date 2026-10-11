# AI Knowledge Engine

A local, lightweight Retrieval-Augmented Generation (RAG) and semantic document retrieval system built using dense sentence embeddings, cosine similarity search, and Ollama.

---

## 1. Project Overview

**AI Knowledge Engine** enables fast, private, and offline-capable search and Q&A over local document collections. It ingests PDF documents, splits them into overlapping word windows, computes dense vector embeddings, and persists the index locally. When queried, it retrieves the most semantically relevant passages and prompts a local LLM to generate precise, grounded answers without sending proprietary data to third-party APIs.

---

## 2. System Architecture & Information Flow

```
PDF Documents (data/)
        │
        ▼
[ingestion/pdf_loader.py]  ───► Extract text page-by-page
        │
        ▼
[core/chunker.py]         ───► Sliding-window text chunks with overlap
        │
        ▼
[core/embeddings.py]      ───► Generate vectors via all-MiniLM-L6-v2
        │
        ▼
[core/vector_store.py]    ───► Save vectors (.npy) & metadata (.json) to storage/
        │
        ▼
[core/semantic_search.py] ◄─── User Query (main.py)
        │
        ▼ (Cosine similarity ranking & threshold filtering)
Top-K Context Passages
        │
        ▼
[core/generator.py]       ───► Ollama local LLM ("phi3")
        │
        ▼
Answer Displayed in Terminal
```

---

## 3. Key Capabilities & Features

- **Robust Ingestion**: Extracts text page-by-page from PDFs while gracefully handling 0-byte or corrupted files.
- **Sliding-Window Chunking**: Configurable word-window splitting with boundary overlap to maintain contextual continuity.
- **Dense Embedding Model**: Powered by HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`) with on-demand lazy loading.
- **Lightweight Vector Store**: Zero external database dependencies; stores dense vectors in NumPy `.npy` format and text entries in UTF-8 `.json`.
- **Thresholded Similarity Search**: Vectorized cosine similarity filtering removes low-confidence matches before synthesis.
- **Local LLM Grounding**: Queries Ollama with explicit strict-context instructions to prevent hallucinations.
- **Automated Resource Downloader**: Standalone `download` utility to verify directories and cache model weights before first run.
- **Comprehensive Test Suite**: Automated unit tests with 100% mocked offline execution.

---

## 4. Environment & System Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Version 3.10, 3.11, 3.12, 3.13, or 3.14
- **[Ollama](https://ollama.com/)**: Installed and running locally (for answer synthesis)

---

## 5. Quick Start Installation Guide

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/ka-rthik-l/AI-Knowledge-Engine-.git
cd AI-Knowledge-Engine-

python -m venv venv
```

Activate the environment:
- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Windows (Command Prompt)**: `.\venv\Scripts\activate.bat`
- **Linux / macOS**: `source venv/bin/activate`

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Pre-Downloading Models & Assets (`download`)

Before running queries, you can verify your environment and pre-cache model weights by executing the `download` helper script:

```bash
python download
```

This utility performs the following checks:
1. Verifies that `./data` and `./storage` directories exist.
2. Downloads and caches the `all-MiniLM-L6-v2` embedding weights from HuggingFace.
3. Verifies local Ollama daemon connectivity and pulls the `phi3` model tag.

---

## 7. Configuration Hyperparameters (`config.py`)

All engine hyperparameters are defined in [`config.py`](file:///c:/Users/karth/Projects/AI-Knowledge-Engine-/config.py) and can be customized via environment variables:

| Parameter | Environment Variable | Default | Purpose |
|---|---|---|---|
| `CHUNK_SIZE` | `CHUNK_SIZE` | `120` | Number of words per text chunk |
| `CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `30` | Number of overlapping words between consecutive chunks |
| `TOP_K` | `TOP_K` | `5` | Maximum number of ranked passages passed to the LLM |
| `THRESHOLD` | `THRESHOLD` | `0.3` | Minimum cosine similarity cutoff score (0.0 to 1.0) |
| `STORE_PREFIX` | `STORE_PREFIX` | `storage/store` | Filepath prefix for persisted vector files |
| `LLM_MODEL` | `LLM_MODEL` | `phi3` | Ollama model identifier used for generative answering |

Example override in PowerShell:
```powershell
$env:TOP_K="3"; $env:THRESHOLD="0.4"; python main.py
```

---

## 8. Running the Application (`main.py`)

Place your `.pdf` documents into the `data/` directory, then start the interactive engine:

```bash
python main.py
```

### Execution Flow:
1. **Index Check**: Checks if saved memory exists in `storage/`. If not, scans `data/` for PDFs, chunks them, computes embeddings, and saves the index to disk.
2. **Interactive Prompt**: Prompts `Ask: ` for your query.
3. **Similarity Retrieval**: Displays the top matching passages with sources and similarity scores.
4. **Answer Generation**: Synthesizes a grounded answer using the local Ollama LLM.

---

## 9. Dependency Management (`requirements.txt`)

Dependencies in [`requirements.txt`](file:///c:/Users/karth/Projects/AI-Knowledge-Engine-/requirements.txt) are grouped into clear sections:
- **NLP & Sentence Transformers**: `sentence-transformers`, `transformers`, `huggingface_hub`
- **Deep Learning Framework**: `torch`, `safetensors`, CUDA accelerators
- **Scientific Computing**: `numpy`, `scipy`, `scikit-learn`
- **Document Parsing**: `pypdf`
- **LLM Client**: `ollama`
- **Networking**: `httpx`, `httpcore`, `anyio`

---

## 10. Automated Test Suite

The project includes unit tests for chunking, persistence, similarity search, PDF loading, and generative prompting. All tests run offline with mock dependencies:

```bash
python -m unittest discover tests
```

Output:
```
Ran 26 tests in 0.10s
OK
```

---

## 11. Project Directory Structure

```
AI-Knowledge-Engine-/
├── config.py             # Centralized settings and environment validation
├── download              # Setup and pre-download helper script
├── main.py               # Interactive CLI search and generation entrypoint
├── requirements.txt      # Categorized dependency requirements
├── README.md             # Project documentation and user guide
├── core/
│   ├── chunker.py         # Sliding-window text chunker
│   ├── embeddings.py      # Lazy-loaded SentenceTransformer embeddings
│   ├── generator.py       # Ollama LLM response synthesis
│   ├── semantic_search.py # Cosine similarity retrieval engine
│   └── vector_store.py    # Persistent NumPy vector and JSON text storage
├── ingestion/
│   └── pdf_loader.py      # Resilient PDF text extraction
├── data/                  # Source PDF documents for indexing
├── storage/               # Saved vector index (.npy) and metadata (.json)
└── tests/                 # Automated unit tests
    ├── test_chunker.py
    ├── test_generator.py
    ├── test_pdf_loader.py
    ├── test_semantic_search.py
    └── test_vector_store.py
```

---

## 12. Troubleshooting & FAQ

- **`ModuleNotFoundError: No module named 'pypdf'` or `sentence_transformers`**:
  Ensure your virtual environment is activated and run `pip install -r requirements.txt`.
- **`Ollama connection refused`**:
  Make sure the Ollama application is running (`ollama serve` or desktop client) and that the model is pulled (`ollama pull phi3`).
- **Index Rebuild**:
  To force a rebuild of the vector index from newly added PDFs, delete the files in `storage/` (`store_vectors.npy` and `store_texts.json`) and rerun `python main.py`.

---

## 13. License

Distributed under the MIT License.
