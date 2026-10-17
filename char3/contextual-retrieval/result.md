(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char3/contextual-retrieval/experiment_3_10.py \
  --mode evaluate \
  --method all \
  --context-model "qwen3.8-27b" \
  --confirm-api-cost
Experiment 3-10 | Contextual Retrieval | independent Qwen API
Config: /mnt/e/ai-agent-book/config.yaml
.env: /mnt/e/ai-agent-book/.env loaded=True
Chat: qwen3.8-27b https://dashscope.aliyuncs.com/compatible-mode/v1
Embedding: qwen3.7-text-embedding-flash dim=1024
Embedding endpoint: https://ws-l84cq4l68pzgx8fq.cn-beijing.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
Rerank: qwen3.7-text-rerank endpoint: https://ws-l84cq4l68pzgx8fq.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
API key: set=yes (secret never shown)
Mode: evaluate method: all
Fixture: /mnt/e/ai-agent-book/char3/contextual-retrieval/official_reference_chunks.json exists=True
Eval: /mnt/e/ai-agent-book/char3/contextual-retrieval/official_retrieval_eval_15.json exists=True
Source documents: 2 | LLM prefix requests up to 22 | source_scope: full
[Prefix 01/22] 宪法_chunk_0 | chars=112 | NEW
[Prefix 02/22] 宪法_chunk_1 | chars=105 | NEW
[Prefix 03/22] 宪法_chunk_10 | chars=153 | NEW
[Prefix 04/22] 宪法_chunk_11 | chars=145 | NEW
[Prefix 05/22] 宪法_chunk_12 | chars=148 | NEW
[Prefix 06/22] 宪法_chunk_13 | chars=146 | NEW
[Prefix 07/22] 宪法_chunk_14 | chars=146 | NEW
[Prefix 08/22] 宪法_chunk_15 | chars=175 | NEW
[Prefix 09/22] 宪法_chunk_16 | chars=114 | NEW
[Prefix 10/22] 宪法_chunk_17 | chars=174 | NEW
[Prefix 11/22] 宪法_chunk_2 | chars=121 | NEW
[Prefix 12/22] 宪法_chunk_3 | chars=124 | NEW
[Prefix 13/22] 宪法_chunk_4 | chars=116 | NEW
[Prefix 14/22] 宪法_chunk_5 | chars=103 | NEW
[Prefix 15/22] 宪法_chunk_6 | chars=100 | NEW
[Prefix 16/22] 宪法_chunk_7 | chars=169 | NEW
[Prefix 17/22] 宪法_chunk_8 | chars=90 | NEW
[Prefix 18/22] 宪法_chunk_9 | chars=192 | NEW
[Prefix 19/22] 检察官法_2019_04_23_chunk_0 | chars=105 | NEW
[Prefix 20/22] 检察官法_2019_04_23_chunk_1 | chars=131 | NEW
[Prefix 21/22] 检察官法_2019_04_23_chunk_2 | chars=140 | NEW
[Prefix 22/22] 检察官法_2019_04_23_chunk_3 | chars=150 | NEW
  [Embedding] document 1/22 complete
  [Embedding] document 2/22 complete
  [Embedding] document 3/22 complete
  [Embedding] document 4/22 complete
  [Embedding] document 5/22 complete
  [Embedding] document 6/22 complete
  [Embedding] document 7/22 complete
  [Embedding] document 8/22 complete
  [Embedding] document 9/22 complete
  [Embedding] document 10/22 complete
  [Embedding] document 11/22 complete
  [Embedding] document 12/22 complete
  [Embedding] document 13/22 complete
  [Embedding] document 14/22 complete
  [Embedding] document 15/22 complete
  [Embedding] document 16/22 complete
  [Embedding] document 17/22 complete
  [Embedding] document 18/22 complete
  [Embedding] document 19/22 complete
  [Embedding] document 20/22 complete
  [Embedding] document 21/22 complete
  [Embedding] document 22/22 complete
  [Embedding] document 1/22 complete
  [Embedding] document 2/22 complete
  [Embedding] document 3/22 complete
  [Embedding] document 4/22 complete
  [Embedding] document 5/22 complete
  [Embedding] document 6/22 complete
  [Embedding] document 7/22 complete
  [Embedding] document 8/22 complete
  [Embedding] document 9/22 complete
  [Embedding] document 10/22 complete
  [Embedding] document 11/22 complete
  [Embedding] document 12/22 complete
  [Embedding] document 13/22 complete
  [Embedding] document 14/22 complete
  [Embedding] document 15/22 complete
  [Embedding] document 16/22 complete
  [Embedding] document 17/22 complete
  [Embedding] document 18/22 complete
  [Embedding] document 19/22 complete
  [Embedding] document 20/22 complete
  [Embedding] document 21/22 complete
  [Embedding] document 22/22 complete
  [Embedding] query 1/15 complete
  [Embedding] query 2/15 complete
  [Embedding] query 3/15 complete
  [Embedding] query 4/15 complete
  [Embedding] query 5/15 complete
  [Embedding] query 6/15 complete
  [Embedding] query 7/15 complete
  [Embedding] query 8/15 complete
  [Embedding] query 9/15 complete
  [Embedding] query 10/15 complete
  [Embedding] query 11/15 complete
  [Embedding] query 12/15 complete
  [Embedding] query 13/15 complete
  [Embedding] query 14/15 complete
  [Embedding] query 15/15 complete

====================================================================================
Experiment 3-10 | Plain vs Contextual Retrieval
Chunks=22 | Queries=15 | Corpus source=LIVE_QWEN_PREFIX_GENERATION_FULL
Method                         R@1    R@3    R@5    MRR
plain_bm25                     0.600 0.867 0.933 0.751
contextual_bm25                0.733 0.867 0.933 0.814
plain_dense                    0.867 1.000 1.000 0.933
contextual_dense               0.933 1.000 1.000 0.967
plain_hybrid                   0.667 0.933 1.000 0.813
contextual_hybrid              0.733 0.933 1.000 0.850
API: {'attempts': 81, 'successes': 81, 'failures': 0, 'by_stage': {'context_prefix': 22, 'embedding_document': 44, 'embedding_query': 15}, 'stage_times': {'context_prefix': {'calls': 22, 'total_ms': 309713.51, 'mean_ms': 14077.89}, 'embedding_document': {'calls': 44, 'total_ms': 14740.18, 'mean_ms': 335.0}, 'embedding_query': {'calls': 15, 'total_ms': 4473.74, 'mean_ms': 298.25}}, 'usage': {'completion_tokens': 12070, 'prompt_tokens': 212908, 'total_tokens': 224978}}
Results: /mnt/e/ai-agent-book/char3/contextual-retrieval/results/exp3_10_20261016_103203_781638
====================================================================================