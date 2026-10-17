(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python -u char3/agentic-rag/experiment_3_8.py \
  --mode evaluate \
  --confirm-api-cost
Experiment 3-8 | standalone ReAct vs non-agentic RAG
Config: /mnt/e/ai-agent-book/config.yaml
.env: /mnt/e/ai-agent-book/.env
LLM: qwen3.7-flash | key_set= True | chat_base= https://dashscope.aliyuncs.com/compatible-mode/v1
Mode: evaluate | KB: offline | corpus: /mnt/e/ai-agent-book/char3/agentic-rag/laws
Search top_k=5, iterations=10, max_searches=5, max_tool_calls=20
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.230 seconds.
Prefix dict has been built successfully.
[KB] 288 docs / 21085 chunks indexed in 15.79s
[1/7] 
[easy_1] 故意伤害致人重伤的，如何处罚？
  non-agentic  searches=1   evidence_recall=100%  cite_valid=3  latency=19565ms stop=single_pass
  agentic      searches=1   evidence_recall=100%  cite_valid=1  latency=17049ms stop=model_finished
[2/7] 
[easy_2] 正当防卫是怎么规定的？
  non-agentic  searches=1   evidence_recall=100%  cite_valid=5  latency=26956ms stop=single_pass
  agentic      searches=1   evidence_recall=100%  cite_valid=0  latency=16638ms stop=model_finished
[3/7] 
[easy_3] 醉酒驾驶机动车如何处罚？
  non-agentic  searches=1   evidence_recall=100%  cite_valid=3  latency=14270ms stop=single_pass
  agentic      searches=1   evidence_recall=100%  cite_valid=2  latency=7418ms stop=model_finished
[4/7] 
[hard_1] 故意杀人罪判几年？
  non-agentic  searches=1   evidence_recall=0%    cite_valid=0  latency=13810ms stop=single_pass
  agentic      searches=0   evidence_recall=0%    cite_valid=0  latency=7222ms stop=model_finished
[5/7] 
[hard_2] 盗窃罪的立案标准是什么？
  non-agentic  searches=1   evidence_recall=0%    cite_valid=0  latency=20063ms stop=single_pass
  agentic      searches=3   evidence_recall=100%  cite_valid=1  latency=21781ms stop=model_finished
[6/7] 
[hard_3] 诈骗罪的量刑标准是什么？
  non-agentic  searches=1   evidence_recall=0%    cite_valid=2  latency=29002ms stop=single_pass
  agentic      searches=3   evidence_recall=100%  cite_valid=4  latency=26580ms stop=model_finished
[7/7] 
[hard_4] 醉酒过失致人重伤且有盗窃前科，应如何量刑？
  non-agentic  searches=1   evidence_recall=33%   cite_valid=1  latency=32605ms stop=single_pass
  agentic      searches=0   evidence_recall=0%    cite_valid=0  latency=35343ms stop=model_finished

================================================================================================
Experiment 3-8 | ReAct Agentic vs non-agentic RAG | REAL LLM tool calls
================================================================================================
Mode               Cases    Evidence recall   Mean searches     Mean latency ms  Valid cite rate
agentic                7              71.4%            1.29             18861.6            57.1%
non-agentic            7              47.6%            1.00             22324.4            71.4%
================================================================================================
Evidence recall measures gold retrieval, NOT legal answer accuracy. All generated answers require evidence review.
Results: /mnt/e/ai-agent-book/char3/agentic-rag/results/exp3_8_20261015_143724_812927