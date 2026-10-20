# Experiment 4-1: Active Tool Discovery

- Suite: demo
- Runtime: local vLLM qwen3_0.6b at http://127.0.0.1:8000/v1
- Token estimator: local_model_chat_template
- Catalog: 30 tools

## Strategy comparison

| Strategy | Selection | Precise | Simulation selection completion | Real completion | Mean schema tokens | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|
| full | 0/8 | 0/8 | 0/8 | 0/8 | 3366 | 1.228 |
| prefilter | 2/8 | 2/8 | 2/8 | 0/8 | 1120.1 | 1.098 |
| discovery | 0/8 | 0/8 | 0/8 | 0/8 | 907 | 0.941 |

## Acceptance checks

- PASS: all_results_machine_readable
- PASS: exact_requested_tasks_and_strategies_recorded
- PASS: catalog_archived_and_hashed
- PASS: same_catalog_across_strategies
- PASS: model_local_base_url
- PASS: all_live_runs_have_model_responses
- PASS: each_completed_task_has_trace
- PASS: all_requested_strategy_runs_present
- PASS: no_context_budget_skips
- PASS: no_uncaught_errors
- PASS: no_mock_claimed_as_real
- PASS: 126_reference_demo_catalog_constructed
- PASS: simulated_results_explicitly_labelled

Gate result: PASS

Demo executes only simulated tool results and is NOT proof of actual task completion.

Important: schema-injection tokens are a prompt-construction metric; actual prompt usage is recorded separately from vLLM responses. A scripted offline test is not evidence of model competence.

## Provenance

- Catalog SHA256: 42a17c5f8350ae2f332ef20dfc6c7a91034c74e57a76e00a8d746efdbec4fcca
- Gzip archive SHA256: e55a6f9ab1a7a3e3c12e3ca19bbdeda118bb437d8a0e0447e822a5751a81e708
- Detailed receipts under runs/ and results.json; inspect missed slots, generic-tool selection, invalid actions, discovery misses, and premature finish.
