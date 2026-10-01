# Hybrid-RAG Empirical Benchmark Results

Evaluated against 8 multi-faceted ground-truth corporate policy test queries.

| Retrieval Strategy | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | Mean Reciprocal Rank (MRR) | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Pure Vector Search (all-MiniLM-L6-v2)** | 62.5% | 87.5% | 87.5% | 0.708 | 104.8 ms |
| **2. Pure Keyword Search (BM25Okapi)** | 75.0% | 75.0% | 100.0% | 0.812 | 4.7 ms |
| **3. Hybrid Search (RRF k=60)** | 75.0% | 87.5% | 100.0% | 0.823 | 61.1 ms |
| **4. Hybrid + Cross-Encoder Re-ranker** | 75.0% | 100.0% | 100.0% | 0.875 | 2508.4 ms |
