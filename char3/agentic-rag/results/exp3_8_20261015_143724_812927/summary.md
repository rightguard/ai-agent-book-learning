# Experiment 3-8 live ReAct vs one-shot RAG

Evidence recall measures retrieval of labelled evidence; not factual/legal correctness.

| Method | Cases | Evidence recall | Mean searches | Mean latency ms | Valid citation rate |
|---|---:|---:|---:|---:|---:|
| agentic | 7 | 71.4% | 1.29 | 18861.6 | 57.1% |
| non-agentic | 7 | 47.6% | 1.00 | 22324.4 | 71.4% |

API attempts: 23; usage: {'completion_tokens': 19909, 'prompt_tokens': 25663, 'total_tokens': 45572}

External sources are untrusted. Judge scores (when requested) are not independent by default.
