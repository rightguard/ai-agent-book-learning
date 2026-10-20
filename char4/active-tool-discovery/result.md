(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# conda activate vllm_wslnda activate vllm_wsl
cd /mnt/e/ai-agent-book

python char4/active-tool-discovery/experiment_4_1.py \
  --suite demo \
  --model qwen3_0.6b \
  --base-url http://127.0.0.1:8000/v1 \
  --tool-set-size 30 \
  --strategies full,prefilter,discovery
Experiment 4-1 | Active Tool Discovery | independent local vLLM
Root: /mnt/e/ai-agent-book
Config: /mnt/e/ai-agent-book/config.yaml
.env: /mnt/e/ai-agent-book/.env loaded=True
Model: qwen3_0.6b location: /mnt/e/ai-agent-book/qwen3_0.6b
vLLM base_url: http://127.0.0.1:8000/v1 api_key: EMPTY
Suite: demo offline/scripted: False embedding: lexical
Output directory: /mnt/e/ai-agent-book/char4/active-tool-discovery/results
[NOTE] Original tools_library.py was not found. Using the independent 126-tool fallback catalog; descriptions differ from the book. For the original catalog, use --catalog-source upstream --upstream-tools-file PATH.
Tool catalog source: embedded_standalone_equivalent
Token counter: local_model_chat_template
Catalog: 30 tools | tasks: 8 | strategies: ['full', 'prefilter', 'discovery']
[NOTE] Reduced-size teaching benchmark; NOT an unmodified 126-tool comparison.
[NOTE] Tool execution = SIMULATED. Selection score cannot prove real task execution.
vLLM model availability: ['qwen3_0.6b']
Ch4 Experiment 4-1:   0%|                                                                                                                                                                                  | 0/24 [00:00<?, ?run/s]
[TASK] repeat=1 id=finance+news strategy=full
      [tool] web_search: OK (simulation)
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=3366 calls=2
Ch4 Experiment 4-1:   4%|███████                                                                                                                                                                   | 1/24 [00:03<01:29,  3.90s/run]
[TASK] repeat=1 id=finance+news strategy=prefilter
      [tool] web_search: OK (simulation)
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1077 calls=2
Ch4 Experiment 4-1:   8%|██████████████▏                                                                                                                                                           | 2/24 [00:04<00:47,  2.17s/run]
[TASK] repeat=1 id=finance+news strategy=discovery
      [discover] need=news source -> ['get_news_by_source', 'search_news', 'google_search', 'get_top_headlines', 'bing_search']
      [discover] need=news source -> ['get_news_by_source', 'search_news', 'google_search', 'get_top_headlines', 'bing_search']
      [discover] need=news source -> ['get_news_by_source', 'search_news', 'google_search', 'get_top_headlines', 'bing_search']
[RESULT] status=loop_detected selection=False precise=False real_complete=False schema_tokens=828 calls=4
Ch4 Experiment 4-1:  12%|█████████████████████▎                                                                                                                                                    | 3/24 [00:05<00:33,  1.60s/run]
[TASK] repeat=1 id=arxiv+download strategy=full
[RESULT] status=too_many_invalid_actions selection=False precise=False real_complete=False schema_tokens=3366 calls=3
Ch4 Experiment 4-1:  17%|████████████████████████████▎                                                                                                                                             | 4/24 [00:07<00:29,  1.46s/run]
[TASK] repeat=1 id=arxiv+download strategy=prefilter
      [tool] arxiv_search: OK (simulation)
      [tool] universal_search: OK (simulation)
      [tool] google_search: OK (simulation)
      [tool] semantic_scholar_search: OK (simulation)
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1108 calls=6
Ch4 Experiment 4-1:  21%|███████████████████████████████████▍                                                                                                                                      | 5/24 [00:08<00:30,  1.62s/run]
[TASK] repeat=1 id=arxiv+download strategy=discovery
      [discover] need=latest research papers in transformer domain -> ['semantic_scholar_search', 'arxiv_search', 'ask_knowledge_base', 'get_top_headlines', 'get_crypto_price']
      [discover] need=latest research papers in transformer domain -> ['semantic_scholar_search', 'arxiv_search', 'ask_knowledge_base', 'get_top_headlines', 'get_crypto_price']
      [discover] need=latest research papers in transformer domain -> ['semantic_scholar_search', 'arxiv_search', 'ask_knowledge_base', 'get_top_headlines', 'get_crypto_price']
[RESULT] status=loop_detected selection=False precise=False real_complete=False schema_tokens=843 calls=4
Ch4 Experiment 4-1:  25%|██████████████████████████████████████████▌                                                                                                                               | 6/24 [00:09<00:24,  1.38s/run]
[TASK] repeat=1 id=github+viz strategy=full
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=3366 calls=2
Ch4 Experiment 4-1:  29%|█████████████████████████████████████████████████▌                                                                                                                        | 7/24 [00:10<00:20,  1.20s/run]
[TASK] repeat=1 id=github+viz strategy=prefilter
      [tool] github_list_contributors: OK (simulation)
      [tool] render_chart: OK (simulation)
[RESULT] status=finished selection=True precise=True real_complete=False schema_tokens=1202 calls=3
Ch4 Experiment 4-1:  33%|████████████████████████████████████████████████████████▋                                                                                                                 | 8/24 [00:12<00:21,  1.37s/run]
[TASK] repeat=1 id=github+viz strategy=discovery
      [discover] need=contributor statistics and chart generation -> ['render_chart', 'universal_search', 'get_current_weather', 'search_pubmed', 'search_news']
      [discover] need=contributor statistics and chart generation -> ['render_chart', 'universal_search', 'get_current_weather', 'search_pubmed', 'search_news']
      [discover] need=contributor statistics and chart generation -> ['render_chart', 'universal_search', 'get_current_weather', 'search_pubmed', 'search_news']
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=823 calls=4
Ch4 Experiment 4-1:  38%|███████████████████████████████████████████████████████████████▊                                                                                                          | 9/24 [00:13<00:18,  1.26s/run]
[TASK] repeat=1 id=weather+calendar strategy=full
[RESULT] status=too_many_invalid_actions selection=False precise=False real_complete=False schema_tokens=3366 calls=3
Ch4 Experiment 4-1:  42%|██████████████████████████████████████████████████████████████████████▍                                                                                                  | 10/24 [00:14<00:15,  1.12s/run]
[TASK] repeat=1 id=weather+calendar strategy=prefilter
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1124 calls=2
Ch4 Experiment 4-1:  46%|█████████████████████████████████████████████████████████████████████████████▍                                                                                           | 11/24 [00:14<00:13,  1.00s/run]
[TASK] repeat=1 id=weather+calendar strategy=discovery
      [discover] need=weather information -> ['web_search', 'get_current_weather', 'get_weather_forecast', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=weather information -> ['web_search', 'get_current_weather', 'get_weather_forecast', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=weather information -> ['web_search', 'get_current_weather', 'get_weather_forecast', 'arxiv_search', 'ask_knowledge_base']
[RESULT] status=loop_detected selection=False precise=False real_complete=False schema_tokens=847 calls=4
Ch4 Experiment 4-1:  50%|████████████████████████████████████████████████████████████████████████████████████▌                                                                                    | 12/24 [00:15<00:11,  1.04run/s]
[TASK] repeat=1 id=forex+weather strategy=full
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=3366 calls=2
Ch4 Experiment 4-1:  54%|███████████████████████████████████████████████████████████████████████████████████████████▌                                                                             | 13/24 [00:16<00:10,  1.07run/s]
[TASK] repeat=1 id=forex+weather strategy=prefilter
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1126 calls=2
Ch4 Experiment 4-1:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                      | 14/24 [00:17<00:09,  1.11run/s]
[TASK] repeat=1 id=forex+weather strategy=discovery
      [discover] need=weather information -> ['web_search', 'get_current_weather', 'get_weather_forecast', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=weather information -> ['web_search', 'get_current_weather', 'get_weather_forecast', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=current time -> ['get_current_weather', 'bing_search', 'get_forex_rate', 'get_news_by_source', 'search_news']
      [discover] need=currency exchange rate -> ['get_forex_rate', 'convert_currency', 'arxiv_search', 'ask_knowledge_base', 'bing_search']
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1455 calls=5
Ch4 Experiment 4-1:  62%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                               | 15/24 [00:18<00:09,  1.02s/run]
[TASK] repeat=1 id=crypto+news strategy=full
      [tool] get_stock_price: OK (simulation)
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=3366 calls=2
Ch4 Experiment 4-1:  67%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                        | 16/24 [00:20<00:09,  1.18s/run]
[TASK] repeat=1 id=crypto+news strategy=prefilter
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=1113 calls=2
Ch4 Experiment 4-1:  71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                 | 17/24 [00:21<00:09,  1.31s/run]
[TASK] repeat=1 id=crypto+news strategy=discovery
      [discover] need=latest information about Ethereum's price and news -> ['get_crypto_price', 'get_stock_price', 'web_search', 'get_market_index', 'get_top_headlines']
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=813 calls=2
Ch4 Experiment 4-1:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                          | 18/24 [00:23<00:07,  1.23s/run]
[TASK] repeat=1 id=opinion(诱导) strategy=full
[RESULT] status=too_many_invalid_actions selection=False precise=False real_complete=False schema_tokens=3366 calls=3
Ch4 Experiment 4-1:  79%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 19/24 [00:24<00:06,  1.24s/run]
[TASK] repeat=1 id=opinion(诱导) strategy=prefilter
      [tool] search_news: OK (simulation)
[RESULT] status=finished selection=True precise=True real_complete=False schema_tokens=1073 calls=2
Ch4 Experiment 4-1:  83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 20/24 [00:25<00:04,  1.23s/run]
[TASK] repeat=1 id=opinion(诱导) strategy=discovery
      [discover] need=news and opinion trends -> ['search_news', 'get_news_by_source', 'bing_search', 'web_search', 'get_top_headlines']
      [discover] need=news and opinion trends -> ['search_news', 'get_news_by_source', 'bing_search', 'web_search', 'get_top_headlines']
      [discover] need=news and opinion trends -> ['search_news', 'get_news_by_source', 'bing_search', 'web_search', 'get_top_headlines']
[RESULT] status=finished selection=False precise=False real_complete=False schema_tokens=831 calls=4
Ch4 Experiment 4-1:  88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                     | 21/24 [00:27<00:03,  1.32s/run]
[TASK] repeat=1 id=academic(诱导) strategy=full
[RESULT] status=too_many_invalid_actions selection=False precise=False real_complete=False schema_tokens=3366 calls=3
Ch4 Experiment 4-1:  92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉              | 22/24 [00:28<00:02,  1.29s/run]
[TASK] repeat=1 id=academic(诱导) strategy=prefilter
[RESULT] status=too_many_invalid_actions selection=False precise=False real_complete=False schema_tokens=1138 calls=3
Ch4 Experiment 4-1:  96%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 23/24 [00:29<00:01,  1.29s/run]
[TASK] repeat=1 id=academic(诱导) strategy=discovery
      [discover] need=quantum computing research advancements -> ['google_search', 'semantic_scholar_search', 'universal_search', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=quantum computing research advancements -> ['google_search', 'semantic_scholar_search', 'universal_search', 'arxiv_search', 'ask_knowledge_base']
      [discover] need=quantum computing research advancements -> ['google_search', 'semantic_scholar_search', 'universal_search', 'arxiv_search', 'ask_knowledge_base']
[RESULT] status=loop_detected selection=False precise=False real_complete=False schema_tokens=816 calls=4
Ch4 Experiment 4-1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:30<00:00,  1.28s/run]

==========================================================================================
Experiment 4-1 | Active Tool Discovery | local vLLM
Suite: demo | Catalog: 30 | Token counter: local_model_chat_template
Strategy        Selection    Precise       Real   SchemaTok(avg)   Latency(avg)
full                  0/8        0/8    0/8               3366.0           1.23
prefilter             2/8        2/8    0/8               1120.1           1.10
discovery             0/8        0/8    0/8                907.0           0.94
Full / active schema tokens ratio: 3.71x
Acceptance gates:
  [PASS] all_results_machine_readable
  [PASS] exact_requested_tasks_and_strategies_recorded
  [PASS] catalog_archived_and_hashed
  [PASS] same_catalog_across_strategies
  [PASS] model_local_base_url
  [PASS] all_live_runs_have_model_responses
  [PASS] each_completed_task_has_trace
  [PASS] all_requested_strategy_runs_present
  [PASS] no_context_budget_skips
  [PASS] no_uncaught_errors
  [PASS] no_mock_claimed_as_real
  [PASS] 126_reference_demo_catalog_constructed
  [PASS] simulated_results_explicitly_labelled
Overall: PASS
Results: /mnt/e/ai-agent-book/char4/active-tool-discovery/results/exp4_1_20261019_105841_843676
==========================================================================================