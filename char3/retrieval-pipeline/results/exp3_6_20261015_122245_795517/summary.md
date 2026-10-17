# Experiment 3-6: stage-by-stage retrieval evaluation

Documents: 20; chunks: 22; queries: 12

| Stage | Recall@3 | Hit@3 | MRR | nDCG@3 | Mean latency (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.8333 | 0.8333 | 0.7917 | 0.8026 | 113.8 |
| Dense | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 346.4 |
| Hybrid-RRF | 1.0000 | 1.0000 | 0.9583 | 0.9692 | 429.6 |
| Hybrid-Weighted | 1.0000 | 1.0000 | 0.9583 | 0.9692 | 429.6 |
| Hybrid-RRF+Rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 801.4 |

Interpretation notes:
- Recall@k is standard multi-gold recall; Hit@k is the upstream book's simplified success measure.
- A reranker does not retrieve unseen documents; the RRF candidate pool bounds its performance.
- This is a small educational 20-document / 12-query benchmark; avoid broad generalizations.
- This single-process implementation does not test microservice API availability.
- Document embedding API calls are one-time indexing cost, separate from per-query latency.
- API receipts and total token usage are recorded without secrets.
- Full traces, component timing, and gold references are in results.json.
