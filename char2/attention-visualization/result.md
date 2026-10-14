(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char2/attention-visualization/experiment_2_2.py --device cuda
本地模型：/mnt/e/ai-agent-book/qwen3_0.6b
设备：cuda
输出目录：/mnt/e/ai-agent-book/char2/attention-visualization/results/20261013_101942_225205
2-2 加载本地模型:  12%|███████████████████████▏                                                                                                                                                                 | 1/8 [00:01<00:10,  1.54s/阶段]`torch_dtype` is deprecated! Use `dtype` instead!
2-2 生成实际推理序列:  62%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                   | 5/8 [00:50<00:24,  8.14s/阶段The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.                                                               | 0/640 [00:00<?, ?token/s]
2-2 真实 Token 生成:  60%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                       | 382/640 [01:00<00:40,  6.33token/s]
2-2 保存推理热力图: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [01:53<00:00, 14.16s/阶段]
结果已保存：/mnt/e/ai-agent-book/char2/attention-visualization/results/20261013_101942_225205

[注意力统计]
  layer_0 | Sink平均占比 14.7% | 因果上三角最大值 0.00e+00
  layer_13 | Sink平均占比 73.7% | 因果上三角最大值 0.00e+00
  layer_27 | Sink平均占比 83.2% | 因果上三角最大值 0.00e+00
[思考区] 368 token | [答案区] 14 token