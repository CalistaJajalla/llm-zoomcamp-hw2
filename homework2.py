"""
LLM Zoomcamp 2026 - Homework 2: Vector Search
Run: uv run python homework2.py
Requires: onnxruntime tokenizers numpy tqdm minsearch gitsource
          + download.py and embedder.py in the same folder
          + model downloaded via: uv run python download.py
"""

import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from embedder import Embedder

# Load embedder
embedder = Embedder()

# Q1: Embed a query 
print("Q1: Embedding a query")
query_ann = "How does approximate nearest neighbor search work?"
v = embedder.encode(query_ann)
print(f"Vector length: {len(v)}")
print(f"v[0] = {v[0]:.4f}")
print(f"Answer: {v[0]:.2f}")

# Load documents 
print("\nLoading lesson pages from GitHub...")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents")

# Q2: Cosine similarity 
print("\nQ2: Cosine similarity")
target_file = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = next(d for d in documents if d["filename"] == target_file)
v_doc = embedder.encode(target_doc["content"])

# Vectors are normalized so dot product = cosine similarity
similarity = float(np.dot(v, v_doc))
print(f"Cosine similarity with {target_file}:")
print(f"Answer: {similarity:.2f}")

# Q3: Chunking and search by hand
print("\nChunking documents...")
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Total chunks: {len(chunks)}")

print("\nQ3: Chunking and search by hand")
print("Embedding all chunks...")
contents = [c["content"] for c in chunks]
X = embedder.encode_batch(contents)
print(f"Embedding matrix shape: {X.shape}")

scores = X.dot(v)
best_idx = int(np.argmax(scores))
print(f"Best chunk index: {best_idx}")
print(f"Best score: {scores[best_idx]:.4f}")
print(f"Answer (filename): {chunks[best_idx]['filename']}")

# Q4: Vector search with minsearch
print("\nQ4: Vector search with minsearch")
vector_index = VectorSearch(keyword_fields=["filename"])
payload = [{"filename": c["filename"], "content": c["content"], "start": c["start"]} for c in chunks]
vector_index.append_batch(X, payload)

query_metrics = "What metric do we use to evaluate a search engine?"
v_metrics = embedder.encode(query_metrics)
vector_results = vector_index.search(v_metrics, num_results=5)

print(f"Query: {query_metrics}")
print("Top 5 results:")
for i, r in enumerate(vector_results):
    print(f"  {i+1}. {r['filename']}")
print(f"Answer (first result): {vector_results[0]['filename']}")

# Q5: Text search vs vector search
print("\nQ5: Text search vs vector search")
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

query_pg = "How do I store vectors in PostgreSQL?"
v_pg = embedder.encode(query_pg)

text_results = text_index.search(query_pg, num_results=5)
vec_results_pg = vector_index.search(v_pg, num_results=5)

text_files = set(r["filename"] for r in text_results)
vec_files = set(r["filename"] for r in vec_results_pg)

print(f"Query: {query_pg}")
print(f"Text top 5:   {[r['filename'] for r in text_results]}")
print(f"Vector top 5: {[r['filename'] for r in vec_results_pg]}")
print(f"In vector but NOT in text: {vec_files - text_files}")

# Q6: Hybrid search (RRF)
print("\nQ6: Hybrid search with RRF")

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query_tools = "How do I give the model access to tools?"
v_tools = embedder.encode(query_tools)

vec_tools = vector_index.search(v_tools, num_results=5)
txt_tools = text_index.search(query_tools, num_results=5)
hybrid = rrf([vec_tools, txt_tools])

print(f"Query: {query_tools}")
print(f"Vector top 5: {[r['filename'] for r in vec_tools]}")
print(f"Text top 5:   {[r['filename'] for r in txt_tools]}")
print(f"Hybrid (RRF) top 5: {[r['filename'] for r in hybrid]}")
print(f"Answer (first after RRF): {hybrid[0]['filename']}")

# Summary of HW Answers
print("\n" + "="*60)
print("HOMEWORK ANSWERS SUMMARY =^-^= ")
print("="*60)
print(f"Q1: v[0] = {v[0]:.2f}")
print(f"Q2: cosine similarity = {similarity:.2f}")
print(f"Q3: best chunk filename = {chunks[best_idx]['filename']}")
print(f"Q4: first vector result = {vector_results[0]['filename']}")
print(f"Q5: in vector not in text = {vec_files - text_files}")
print(f"Q6: first after RRF = {hybrid[0]['filename']}")
