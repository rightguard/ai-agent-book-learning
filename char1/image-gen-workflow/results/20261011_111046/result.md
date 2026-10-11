
(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char1/image-gen-workflow/image_gen.py
图像API请求地址： https://ws-xxxxxx.maas.aliyuncs.com/compatible-mode/v1/images/generations（此处屏蔽真实空间id）
项目根目录：/mnt/e/ai-agent-book
配置文件：/mnt/e/ai-agent-book/config.yaml
环境文件：/mnt/e/ai-agent-book/.env
文本模型：qwen3.7-flash-2026-07-15
图像模型：qwen-image-3.0-pro
输出目录：/mnt/e/ai-agent-book/char1/image-gen-workflow/results
实验任务数：6
实验1-4·图片生成总进度:   0%|                    | 0/6 [00:00<?, ?it/s, weekend_programmer | direct]
[完成] weekend_programmer | 原始需求直出的图 -> images/01_weekend_programmer_direct.png
实验1-4·图片生成总进度:  17%|█▌       | 1/6 [00:11<00:55, 11.19s/it, weekend_programmer | rewritten]
[完成] weekend_programmer | 改写后生成的图 -> images/02_weekend_programmer_rewritten.png
实验1-4·图片生成总进度:  33%|████▋         | 2/6 [00:23<00:48, 12.13s/it, headphone_poster | direct]
[完成] headphone_poster | 原始需求直出的图 -> images/03_headphone_poster_direct.png
实验1-4·图片生成总进度:  50%|█████▌     | 3/6 [00:33<00:32, 10.85s/it, headphone_poster | rewritten]
[完成] headphone_poster | 改写后生成的图 -> images/04_headphone_poster_rewritten.png
实验1-4·图片生成总进度:  67%|██████████▋     | 4/6 [00:46<00:23, 11.59s/it, agi_programmer | direct]
[完成] agi_programmer | 原始需求直出的图 -> images/05_agi_programmer_direct.png
实验1-4·图片生成总进度:  83%|██████████▊  | 5/6 [00:55<00:10, 10.84s/it, agi_programmer | rewritten]
[完成] agi_programmer | 改写后生成的图 -> images/06_agi_programmer_rewritten.png
实验1-4·图片生成总进度: 100%|█████████████| 6/6 [01:07<00:00, 11.29s/it, agi_programmer | rewritten]

[结果汇总]
  成功：6 / 6
  失败：0
  结果目录：/mnt/e/ai-agent-book/char1/image-gen-workflow/results/20261011_111046
  Manifest：/mnt/e/ai-agent-book/char1/image-gen-workflow/results/20261011_111046/manifest.json
