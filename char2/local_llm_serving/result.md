(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" vllm serve /mnt/e/ai-agent-book/qwen3_0.6b   --served-model-name qwen3_0.6b   --host 127.0.0.1   --port 8000   --enable-auto-tool-choice   --tool-call-parser hermes   --gpu-memory-utilization 0.50   --max-model-len 6144   --max-num-seqs 8   --max-num-batched-tokens 1024   --enable-chunked-prefill   --enforce-eager
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299] 
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299]        █     █     █▄   ▄█
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.19.0
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299]   █▄█▀ █     █     █     █  model   /mnt/e/ai-agent-book/qwen3_0.6b
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:299] 
(APIServer pid=629) INFO 10-13 10:05:56 [utils.py:233] non-default args: {'model_tag': '/mnt/e/ai-agent-book/qwen3_0.6b', 'enable_auto_tool_choice': True, 'tool_call_parser': 'hermes', 'host': '127.0.0.1', 'model': '/mnt/e/ai-agent-book/qwen3_0.6b', 'max_model_len': 6144, 'enforce_eager': True, 'served_model_name': ['qwen3_0.6b'], 'gpu_memory_utilization': 0.5, 'max_num_batched_tokens': 1024, 'max_num_seqs': 8, 'enable_chunked_prefill': True}
(APIServer pid=629) INFO 10-13 10:05:56 [model.py:549] Resolved architecture: Qwen3ForCausalLM
(APIServer pid=629) INFO 10-13 10:05:56 [model.py:1678] Using max model len 6144
(APIServer pid=629) INFO 10-13 10:05:56 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=1024.
(APIServer pid=629) INFO 10-13 10:05:56 [vllm.py:790] Asynchronous scheduling is enabled.
(APIServer pid=629) WARNING 10-13 10:05:56 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(APIServer pid=629) WARNING 10-13 10:05:56 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=629) INFO 10-13 10:05:56 [vllm.py:1025] Cudagraph is disabled under eager mode
(APIServer pid=629) INFO 10-13 10:05:56 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=693) INFO 10-13 10:06:13 [core.py:105] Initializing a V1 LLM engine (v0.19.0) with config: model='/mnt/e/ai-agent-book/qwen3_0.6b', speculative_config=None, tokenizer='/mnt/e/ai-agent-book/qwen3_0.6b', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=6144, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=qwen3_0.6b, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [1024], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
(EngineCore pid=693) WARNING 10-13 10:06:13 [interface.py:525] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
(EngineCore pid=693) INFO 10-13 10:06:13 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.25.13.113:47893 backend=nccl
(EngineCore pid=693) INFO 10-13 10:06:13 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
(EngineCore pid=693) INFO 10-13 10:06:15 [gpu_model_runner.py:4735] Starting to load model /mnt/e/ai-agent-book/qwen3_0.6b...
(EngineCore pid=693) INFO 10-13 10:06:16 [cuda.py:334] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(EngineCore pid=693) INFO 10-13 10:06:16 [flash_attn.py:596] Using FlashAttention version 2
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:18<00:00, 18.66s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:18<00:00, 18.66s/it]
(EngineCore pid=693) 
(EngineCore pid=693) INFO 10-13 10:06:36 [default_loader.py:384] Loading weights took 19.21 seconds
(EngineCore pid=693) INFO 10-13 10:06:36 [gpu_model_runner.py:4820] Model loading took 1.12 GiB memory and 20.811041 seconds
(EngineCore pid=693) INFO 10-13 10:06:56 [gpu_worker.py:436] Available KV cache memory: 1.8 GiB
(EngineCore pid=693) INFO 10-13 10:06:56 [kv_cache_utils.py:1319] GPU KV cache size: 16,864 tokens
(EngineCore pid=693) INFO 10-13 10:06:56 [kv_cache_utils.py:1324] Maximum concurrency for 6,144 tokens per request: 2.74x
(EngineCore pid=693) INFO 10-13 10:06:56 [core.py:283] init engine (profile, create kv cache, warmup model) took 19.64 seconds
(EngineCore pid=693) INFO 10-13 10:06:58 [vllm.py:790] Asynchronous scheduling is enabled.
(EngineCore pid=693) WARNING 10-13 10:06:58 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=693) WARNING 10-13 10:06:58 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=693) INFO 10-13 10:06:58 [vllm.py:1025] Cudagraph is disabled under eager mode
(EngineCore pid=693) INFO 10-13 10:06:58 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=629) WARNING 10-13 10:06:58 [interface.py:525] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
(APIServer pid=629) INFO 10-13 10:06:58 [api_server.py:590] Supported tasks: ['generate']
(APIServer pid=629) INFO 10-13 10:06:58 [parser_manager.py:202] "auto" tool choice has been enabled.
(APIServer pid=629) WARNING 10-13 10:06:58 [model.py:1435] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=629) INFO 10-13 10:06:59 [hf.py:314] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(APIServer pid=629) INFO 10-13 10:07:00 [api_server.py:594] Starting vLLM server on http://127.0.0.1:8000
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:37] Available routes are:
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /docs, Methods: HEAD, GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=629) INFO 10-13 10:07:00 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=629) INFO:     Started server process [629]
(APIServer pid=629) INFO:     Waiting for application startup.
(APIServer pid=629) INFO:     Application startup complete.
(APIServer pid=629) INFO:     127.0.0.1:49130 - "GET /v1/models HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:41248 - "GET /v1/models HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:41250 - "GET /v1/models HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:41266 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:10:00 [loggers.py:259] Engine 000: Avg prompt throughput: 35.6 tokens/s, Avg generation throughput: 8.4 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.7%, Prefix cache hit rate: 0.0%
(APIServer pid=629) INFO 10-13 10:10:10 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 16.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.6%, Prefix cache hit rate: 0.0%
(APIServer pid=629) INFO:     127.0.0.1:36880 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:10:20 [loggers.py:259] Engine 000: Avg prompt throughput: 17.4 tokens/s, Avg generation throughput: 15.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.6%, Prefix cache hit rate: 56.5%
(APIServer pid=629) INFO 10-13 10:10:30 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 14.3 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 56.5%
(APIServer pid=629) INFO:     127.0.0.1:53282 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:10:40 [loggers.py:259] Engine 000: Avg prompt throughput: 0.4 tokens/s, Avg generation throughput: 1.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.3%, Prefix cache hit rate: 66.1%
(APIServer pid=629) INFO 10-13 10:10:50 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 16.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.2%, Prefix cache hit rate: 66.1%
(APIServer pid=629) INFO:     127.0.0.1:49364 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:11:00 [loggers.py:259] Engine 000: Avg prompt throughput: 16.9 tokens/s, Avg generation throughput: 14.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.1%, Prefix cache hit rate: 70.7%
(APIServer pid=629) INFO 10-13 10:11:10 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 15.7 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 6.0%, Prefix cache hit rate: 70.7%
(APIServer pid=629) INFO:     127.0.0.1:55438 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:11:20 [loggers.py:259] Engine 000: Avg prompt throughput: 4.3 tokens/s, Avg generation throughput: 16.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.1%, Prefix cache hit rate: 69.5%
(APIServer pid=629) INFO:     127.0.0.1:59946 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:11:30 [loggers.py:259] Engine 000: Avg prompt throughput: 1.1 tokens/s, Avg generation throughput: 18.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 69.5%
(APIServer pid=629) INFO 10-13 10:11:40 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 16.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 69.5%
(APIServer pid=629) INFO:     127.0.0.1:60246 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:11:50 [loggers.py:259] Engine 000: Avg prompt throughput: 1.1 tokens/s, Avg generation throughput: 17.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.1%, Prefix cache hit rate: 69.6%
(APIServer pid=629) INFO:     127.0.0.1:42102 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:12:00 [loggers.py:259] Engine 000: Avg prompt throughput: 1.1 tokens/s, Avg generation throughput: 16.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.7%, Prefix cache hit rate: 69.7%
(APIServer pid=629) INFO 10-13 10:12:10 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 16.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.6%, Prefix cache hit rate: 69.7%
(APIServer pid=629) INFO:     127.0.0.1:46798 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:12:20 [loggers.py:259] Engine 000: Avg prompt throughput: 1.1 tokens/s, Avg generation throughput: 15.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.9%, Prefix cache hit rate: 69.8%
(APIServer pid=629) INFO:     127.0.0.1:50974 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:12:30 [loggers.py:259] Engine 000: Avg prompt throughput: 1.1 tokens/s, Avg generation throughput: 16.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 69.9%
(APIServer pid=629) INFO 10-13 10:12:40 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 15.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.4%, Prefix cache hit rate: 69.9%
(APIServer pid=629) INFO:     127.0.0.1:45868 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:12:50 [loggers.py:259] Engine 000: Avg prompt throughput: 3.5 tokens/s, Avg generation throughput: 15.3 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 68.9%
(APIServer pid=629) INFO:     127.0.0.1:43570 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:43572 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:13:00 [loggers.py:259] Engine 000: Avg prompt throughput: 2.2 tokens/s, Avg generation throughput: 23.1 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.4%, Prefix cache hit rate: 68.9%
(APIServer pid=629) INFO:     127.0.0.1:52592 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:52598 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:52600 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:52604 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:13:10 [loggers.py:259] Engine 000: Avg prompt throughput: 4.4 tokens/s, Avg generation throughput: 56.2 tokens/s, Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.5%, Prefix cache hit rate: 68.9%
(APIServer pid=629) INFO:     127.0.0.1:57262 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57270 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57290 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57282 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57300 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57306 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57322 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57330 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:57342 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:13:20 [loggers.py:259] Engine 000: Avg prompt throughput: 284.1 tokens/s, Avg generation throughput: 105.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 16.4%, Prefix cache hit rate: 36.9%
(APIServer pid=629) INFO:     127.0.0.1:47954 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:47970 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:47982 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:47998 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:48012 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:48028 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:48030 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:48044 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:48054 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:50418 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO:     127.0.0.1:50428 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=629) INFO 10-13 10:13:30 [loggers.py:259] Engine 000: Avg prompt throughput: 1104.0 tokens/s, Avg generation throughput: 16.7 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 12.2%, Prefix cache hit rate: 51.6%
(APIServer pid=629) INFO 10-13 10:13:40 [loggers.py:259] Engine 000: Avg prompt throughput: 275.9 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 51.6%
(APIServer pid=629) INFO 10-13 10:13:50 [loggers.py:259] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 51.6%

(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char2/local_llm_serving/experiment_2_1.py
[本地模型身份] 已核验：True | 摘要：未返回
[实验 2-1] 模型：qwen3_0.6b / 后端：vllm / 输出：/mnt/e/ai-agent-book/char2/local_llm_serving/results/20261013_100951_849137

--- 2-1: tools ---
2-1 ReAct 循环:   0%|                                                                                                                                                                                                     | 0/4 [00:00<?, ?轮/s]
[模型回答] <think>
Okay, the user is asking for the current time and weather in Vancouver. I need to call both functions: get_current_time and get_current_temperature. Let me check the tools provided.

First, for the time, the function get_current_time requires the timezone parameter, which is America/Vancouver. So I'll use that. Then for the temperature, the function get_current_temperature needs the location as Vancouver, Canada, and the unit. The user didn't specify the unit, but the function allows celsius or fahrenheit. Since the user didn't mention, maybe I should default to celsius? Wait, the tool's parameters say unit is an enum, so I can choose. But the user might not care, so maybe just use celsius. Let me make sure to include both parameters in the tool calls.

So, first tool call for time: name get_current_time, arguments {"timezone": "America/Vancouver"}. Second tool call for temperature: name get_current_temperature, arguments {"location": "Vancouver, Canada", "unit": "celsius"} or maybe fahrenheit. Since the user didn't specify, perhaps use celsius. Let me check the function's required parameters again. The location is required, so I need to include that. The unit is optional but enum. So I'll include both parameters. Alright, that should cover both the time and the temperature.
</think>


2-1 ReAct 循环:  25%|███████████████████████████████████████████████▎                                                                                                                                             | 1/4 [00:23<01:11, 23.77s/轮][第 1 轮工具请求] ['get_current_time', 'get_current_temperature']
  [工具反馈] get_current_time: {"timezone": "America/Vancouver", "datetime": "2026-10-12T19:10:15.742837-07:00"}
  [工具反馈] get_current_temperature: {"location": "Vancouver, Canada", "temperature": 10.2, "unit": "°C", "observed_at": "2026-10-12T19:00", "source": "https://api.open-meteo.com/v1/forecast?latitude=49.2827&longitude=-123.1207&current=temperature_2m&timezone=America%2FVancouv

[模型回答] <think>
Okay, let me put this together. The user asked for the current time and weather in Vancouver. I called both functions. The time response came back as 2026-10-12T19:10:15.742837-07:00, which is 7:10 PM. The temperature is 10.2°C, observed at 19:00. I need to present this clearly. Make sure to mention both the time and temperature, and note the source if needed. Also, check if there's any error in the data, but since the tool provided the data, it's all good. Just format the answer concisely.
</think>

The current time in Vancouver is 7:10 PM (2026-10-12T19:10:15.742837-07:00), and the current temperature is 10.2°C.
2-1 ReAct 循环:  50%|██████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                              | 2/4 [00:37<00:37, 18.58s/轮]
[tools] 完成

--- 2-1: raw ---

[原始模型输出 / 第一轮]
2-1 vllm 原始流: 1片段 [00:00,  2.87片段/s]<think>
2-1 vllm 原始流: 39片段 [00:02, 15.84片段/s].tools provided

2-1 vllm 原始流: 166片段 [00:10, 17.79片段/s].required parameters

2-1 vllm 原始流: 244片段 [00:15, 14.59片段/s] answer.seneesngture
2-1 vllm 原始流: 246片段 [00:15, 14.61片段/s]</think>

2-1 vllm 原始流: 248片段 [00:15, 14.99片段/s]<tool_call>
2-1 vllm 原始流: 268片段 [00:16, 15.77片段/s]"}}ncouvers
2-1 vllm 原始流: 270片段 [00:17, 15.51片段/s]
2-1 vllm 原始流: 272片段 [00:17, 14.43片段/s]
2-1 vllm 原始流: 298片段 [00:18, 17.16片段/s]elsius"}}onmperature
2-1 vllm 原始流: 300片段 [00:19, 15.75片段/s]</tool_call>


[原始模型输出 / 回灌工具后第二轮]
2-1 vllm 原始流: 1片段 [00:00,  8.12片段/s]<think>
2-1 vllm 原始流: 148片段 [00:09, 17.22片段/s] temperature.dee
2-1 vllm 原始流: 150片段 [00:09, 17.80片段/s]</think>

2-1 vllm 原始流: 209片段 [00:13, 15.79片段/s]°C. rent temperature

[raw] 完成

--- 2-1: throughput ---
2-1 流式吞吐 / 首 Token 延迟: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:32<00:00, 15.47s/请求]
[throughput] 完成

--- 2-1: batching ---
2-1 不同并发度批处理: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:33<00:00,  2.26s/请求]
[batching] 完成

--- 2-1: cache ---
2-1 前缀缓存命中/未命中: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:12<00:00,  1.00s/请求]
[cache] 完成

[结果汇总] {"status": "complete", "gates": {"tools": true, "raw": true, "throughput": true, "batching": true, "cache": true}}
[输出] /mnt/e/ai-agent-book/char2/local_llm_serving/results/20261013_100951_849137