# image-gen-workflow — 提示词改写 vs 直出对照（实验 1-4）

同一批中文"文生图"需求，走两条路线各出一张图：

- **direct**：原始中文需求直接丢给图像模型
- **rewritten**：先用文本模型把中文需求改写成"面向图像生成模型的高质量中文 prompt"（补构图/光线/材质/镜头/背景细节），再给图像模型看

对照两类输出看"中间加一层提示词改写"到底有没有价值。

## 文件

| 文件 | 说明 |
|------|------|
| `image_gen.py` | 主脚本：读 `config.yaml` → 对每个 case、每条 route 执行 → 下载图片 → 存 manifest/summary |
| `results/<run_id>/` | 每次运行单独一个目录，含 `images/*.png`、`manifest.json`（每张图的完整请求记录）、`summary.json`（索引） |

## 运行

```bash
# 先看任务列表，不消耗 API
python char1/image-gen-workflow/image_gen.py --dry-run

# 只跑某个 case、某条路线
python char1/image-gen-workflow/image_gen.py --case weekend_programmer --route rewritten

# 全量跑（默认）
python char1/image-gen-workflow/image_gen.py
```

`config.yaml` 里 `experiment_1_4.cases` 定义了要跑哪些中文需求；`image` 节点定义图像模型名称、尺寸、seed、是否关闭自动扩写等。

## 2026-10-11 运行结果

| case | 路线 | 文件 |
|------|------|------|
| weekend_programmer | direct | images/01_weekend_programmer_direct.png |
| weekend_programmer | rewritten | images/02_weekend_programmer_rewritten.png |
| headphone_poster | direct | images/03_headphone_poster_direct.png |
| headphone_poster | rewritten | images/04_headphone_poster_rewritten.png |
| agi_programmer | direct | images/05_agi_programmer_direct.png |
| agi_programmer | rewritten | images/06_agi_programmer_rewritten.png |

6 张图全部生成成功（6/6），每张 11 秒左右。完整输出和汇总 JSON 在 `results/20261011_111046/` 下；终端日志见同目录的 `result.md`。

## 关键设计

- **prompt_extend=False**：请求里显式关了图像模型自己的自动提示词扩写，让"直出"这一组和"改写"这一组的差异只来自我们写的中间步骤，不让图像模型那边再偷偷加一层变量。
- **sha256**：每张图下载后算一遍 hash，存进 manifest；如果想做同 prompt 多次出图的一致性测试可以直接对比 hash。
- **响应兼容**：图像模型可能返回 b64_json 也可能返回 url，脚本统一转换成 bytes 再落盘。
