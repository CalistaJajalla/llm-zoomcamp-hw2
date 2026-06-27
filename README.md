# LLM Zoomcamp 2026 - Homework 2: Vector Search

Solution for Module 2 homework of [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp/) by DataTalksClub.

## What this covers

- Embedding text with a lightweight ONNX model (all-MiniLM-L6-v2)
- Computing cosine similarity between vectors using numpy
- Building vector search from scratch with a dot product matrix
- Using minsearch VectorSearch for vector search with chunking
- Comparing keyword vs vector search results
- Combining both with hybrid search using Reciprocal Rank Fusion (RRF)

## Setup

**1. Install uv** (if not already installed)

Mac/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone and install dependencies**

```bash
git clone https://github.com/CalistaJajalla/llm-zoomcamp-hw2.git
cd llm-zoomcamp-hw2
uv sync
```

**3. Download helper scripts**

```bash
curl -O https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed/download.py
curl -O https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed/embedder.py
```

**4. Download the ONNX model**

```bash
uv run python download.py
```

**5. Run**

```bash
uv run python homework2.py
```

## Files

| File | Purpose |
|------|---------|
| `homework2.py` | Main solution script - runs all 6 questions |
| `embedder.py` | Embedder class wrapping the ONNX model |
| `download.py` | Downloads all-MiniLM-L6-v2 from HuggingFace |
| `pyproject.toml` | Project dependencies |

## Answers

| Q | Question | Answer |
|---|----------|--------|
| Q1 | First value of query embedding v[0]? | **-0.02** |
| Q2 | Cosine similarity with sqlitesearch-vector.md? | **0.68** |
| Q3 | Filename of highest-scoring chunk? | **02-vector-search/lessons/09-onnx-embedder.md** |
| Q4 | First result from VectorSearch? | **04-evaluation/lessons/05-search-metrics.md** |
| Q5 | In vector but not in text results? | **02-vector-search/lessons/08-pgvector.md** |
| Q6 | First file after RRF hybrid search? | **01-agentic-rag/lessons/13-function-calling.md** |
