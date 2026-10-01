import os
import time
import pickle
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

# 1. Initialize Clients & Models
print("Loading vector store, BM25 index, and Cross-Encoder model...")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_collection(name="support_docs", embedding_function=sentence_transformer_ef)

with open(BM25_PATH, 'rb') as f:
    bm25 = pickle.load(f)
with open(CHUNKS_PATH, 'rb') as f:
    all_chunks = pickle.load(f)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# 2. Define Search Strategies
def search_vector(query, top_k=20):
    res = collection.query(query_texts=[query], n_results=top_k)
    return res['documents'][0] if res['documents'] else []

def search_bm25(query, top_k=20):
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [all_chunks[i] for i in top_indices]

def search_hybrid_rrf(query, top_k=20, k_rrf=60):
    vec_chunks = search_vector(query, top_k=top_k)
    bm25_chunks = search_bm25(query, top_k=top_k)
    
    rrf_scores = {}
    for rank, chunk in enumerate(vec_chunks):
        rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (k_rrf + rank + 1)
        
    for rank, chunk in enumerate(bm25_chunks):
        rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + 1.0 / (k_rrf + rank + 1)
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:top_k]]

def search_hybrid_crossencoder(query, top_k_retrieval=20, top_n_rerank=5):
    candidate_chunks = search_hybrid_rrf(query, top_k=top_k_retrieval)
    if not candidate_chunks:
        return []
    
    pairs = [[query, chunk] for chunk in candidate_chunks]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_n_rerank]]

# 3. Ground-Truth Benchmark Test Suite (Representative policy queries)
BENCHMARK_DATASET = [
    {
        "query": "What are the documentation requirements under Section V.I.2.b?",
        "ground_truth_keywords": ["V.I.2.b", "Documentation Requirements"]
    },
    {
        "query": "What is the maximum daily lodging rate cap for CONUS travel under 30 days?",
        "ground_truth_keywords": ["$333", "333 per night"]
    },
    {
        "query": "What is the policy exception for prearranged conference lodging exceeding the cap?",
        "ground_truth_keywords": ["conference hotel", "prearranged conference lodging rate"]
    },
    {
        "query": "What is the lodging reimbursement limit for domestic travel assignments of 30 days or more?",
        "ground_truth_keywords": ["100% of the applicable federal per diem rate", "30 days or more"]
    },
    {
        "query": "What are the rules regarding personal automobile mileage and the 300 mile per day standard?",
        "ground_truth_keywords": ["300 mile", "Automobile", "Section V.D.3"]
    },
    {
        "query": "How is lodging or subsistence adjusted when provided without charge or complimentary?",
        "ground_truth_keywords": ["Adjustment for Subsistence or Lodging Provided without Charge", "without charge"]
    },
    {
        "query": "What is the policy for payment of group subsistence expenses?",
        "ground_truth_keywords": ["Payment of Group Subsistence Expenses", "Group Subsistence"]
    },
    {
        "query": "What is the reimbursement procedure for intercampus travel expenses between UC locations?",
        "ground_truth_keywords": ["Intercampus Travel Expenses", "Intercampus"]
    }
]

# 4. Evaluation Engine
def evaluate_strategy(name, search_fn, is_reranked=False):
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    mrr_sum = 0.0
    latencies = []
    
    for item in BENCHMARK_DATASET:
        query = item["query"]
        keywords = item["ground_truth_keywords"]
        
        t0 = time.perf_counter()
        top_chunks = search_fn(query)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        
        # Check rank of first hit
        hit_rank = None
        for rank, chunk in enumerate(top_chunks):
            # Check if any ground truth keyword is present in chunk
            if any(kw.lower() in chunk.lower() for kw in keywords):
                hit_rank = rank + 1
                break
                
        if hit_rank is not None:
            if hit_rank <= 1:
                hit_1 += 1
            if hit_rank <= 3:
                hit_3 += 1
            if hit_rank <= 5:
                hit_5 += 1
            mrr_sum += 1.0 / hit_rank

    total = len(BENCHMARK_DATASET)
    return {
        "strategy": name,
        "hit_rate_1": (hit_1 / total) * 100.0,
        "hit_rate_3": (hit_3 / total) * 100.0,
        "hit_rate_5": (hit_5 / total) * 100.0,
        "mrr": mrr_sum / total,
        "avg_latency_ms": np.mean(latencies)
    }

def run_benchmark():
    print("\n" + "="*80)
    print("RUNNING HYBRID-RAG RETRIEVAL BENCHMARK EVALUATION")
    print(f"Total Benchmark Queries: {len(BENCHMARK_DATASET)}")
    print("="*80 + "\n")
    
    strategies = [
        ("1. Pure Vector Search (all-MiniLM-L6-v2)", lambda q: search_vector(q, top_k=5)),
        ("2. Pure Keyword Search (BM25Okapi)", lambda q: search_bm25(q, top_k=5)),
        ("3. Hybrid Search (RRF k=60)", lambda q: search_hybrid_rrf(q, top_k=5)),
        ("4. Hybrid + Cross-Encoder Re-ranker", lambda q: search_hybrid_crossencoder(q, top_k_retrieval=20, top_n_rerank=5))
    ]
    
    results = []
    for name, fn in strategies:
        print(f"Benchmarking: {name}...")
        res = evaluate_strategy(name, fn)
        results.append(res)
        
    print("\n" + "="*80)
    print(f"{'Retrieval Strategy':<42} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<6} | {'Avg Latency'}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<42} | {r['hit_rate_1']:>5.1f}% | {r['hit_rate_3']:>5.1f}% | {r['hit_rate_5']:>5.1f}% | {r['mrr']:>5.3f} | {r['avg_latency_ms']:>6.1f} ms")
    print("="*80 + "\n")
    
    # Save results to markdown
    md_content = "# Hybrid-RAG Empirical Benchmark Results\n\n"
    md_content += f"Evaluated against {len(BENCHMARK_DATASET)} multi-faceted ground-truth corporate policy test queries.\n\n"
    md_content += "| Retrieval Strategy | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | Mean Reciprocal Rank (MRR) | Avg Latency (ms) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for r in results:
        md_content += f"| **{r['strategy']}** | {r['hit_rate_1']:.1f}% | {r['hit_rate_3']:.1f}% | {r['hit_rate_5']:.1f}% | {r['mrr']:.3f} | {r['avg_latency_ms']:.1f} ms |\n"
        
    with open(os.path.join(BASE_DIR, "benchmark_results.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    print("[OK] Benchmark results saved to benchmark_results.md")

if __name__ == "__main__":
    run_benchmark()
