
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_MODEL_DIR = Path("/mnt/e/ai-agent-book/qwen3_0.6b")


def validate_local_model_dir(raw: str | Path) -> Path:
    """显式检查本地 HF 目录，防止模型名被误认为 Hub ID。"""
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"本地模型目录不存在：{path}\n"
            f"请在 WSL 中确认：ls -lah {path}\n"
            "--model 必须指向包含 config.json、tokenizer 和模型权重的目录，不能直接指向 model.safetensors。"
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json：{path / 'config.json'}")
    if not (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))):
        raise FileNotFoundError(f"模型目录缺少 .safetensors / pytorch_model*.bin 权重文件：{path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise FileNotFoundError(f"模型目录缺少 tokenizer.json 等分词器文件：{path}")
    return path.resolve()


def project_root() -> Path:
    f = Path(__file__).resolve()
    for root in [f.parent, *f.parents]:
        if (root / "config.yaml").is_file():
            return root
    return Path.cwd()


def read_config(root: Path) -> dict:
    try:
        import yaml
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("请先安装配置依赖：python -m pip install pyyaml python-dotenv") from exc
    load_dotenv(root / ".env")
    path = root / "config.yaml"
    if not path.is_file():
        return {}
    obj = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    return obj.get("experiment_2_2", {}) or {}


def sha256_file(path: Path) -> str:
    """分块计算哈希，避免将一个 1GB 级模型权重整体读入内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def weight_files(model_dir: Path) -> list[Path]:
    return sorted(list(model_dir.glob('*.safetensors')) + list(model_dir.glob('pytorch_model*.bin')))


def local_model_fingerprint(model_dir: Path, include_weights: bool = True) -> dict:
    """记录真正加载的本地模型文件身份，而不是猜测不可获得的 HF commit。"""
    candidates = [model_dir / file for file in
                  ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'generation_config.json', 'model.safetensors.index.json')]
    candidates.extend(weight_files(model_dir) if include_weights else [])
    candidates = [file for file in candidates if file.is_file()]
    return {
        'model_dir': str(model_dir),
        'weights_sha256_recorded': include_weights,
        'files': {file.name: {'size_bytes': file.stat().st_size,
                              'sha256': sha256_file(file)} for file in candidates},
    }



def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def resolve_layers(raw: str, count: int) -> list[int]:
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        raise ValueError("--layers 不能为空")
    indices = []
    for part in parts:
        ix = int(part)
        ix = ix if ix >= 0 else count + ix
        if not 0 <= ix < count:
            raise ValueError(f"层索引 {part} 越界；模型共有 {count} 层")
        if ix not in indices:
            indices.append(ix)
    return indices


def token_strings(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False) for tid in ids]


def capture_attention(model, input_ids, layers: list[int], head: int = -1) -> tuple[dict, int, int]:
    """真正前向推理，读取 output_attentions=True 张量，而不是推测/模拟数据。"""
    import torch

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)
    attns = getattr(outputs, "attentions", None)
    if not attns or not all(a is not None for a in attns):
        raise RuntimeError("没有获取到注意力张量，请升级 Transformers 并确保 attn_implementation='eager'")
    layer_count = len(attns)
    head_count = int(attns[0].shape[1])
    if head >= head_count:
        raise ValueError(f"注意力头 --head {head} 超出范围 0~{head_count-1}")
    if head < -1:
        raise ValueError("--head 只能为 -1（平均）或非负注意力头索引")
    matrices = {}
    for index in layers:
        tensor = attns[index][0]  # 形状 [head_count, tokens, tokens]
        picked = tensor.float().mean(dim=0) if head == -1 else tensor[head].float()
        matrices[f"layer_{index}"] = picked.detach().cpu().numpy().copy()
    del outputs, attns
    return matrices, layer_count, head_count


def layer_metrics(matrix) -> dict:
    """查询行（query）对键列（key）的注意力；未来位置应被因果掩码遮挡。"""
    import numpy as np

    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("Attention 矩阵必须是 [tokens, tokens] 的方阵")
    upper = matrix[np.triu_indices(n, k=1)]
    rowsum = matrix.sum(axis=1)
    late_rows = np.arange(n // 2, n)
    thirds = np.array_split(np.arange(n), 3)
    thirds_mass = {}
    for name, indexes in zip(("beginning", "middle", "end"), thirds):
        if len(indexes):
            thirds_mass[name] = float(matrix[np.ix_(late_rows, indexes)].sum() / max(1, len(late_rows)))
        else:
            thirds_mass[name] = 0.0
    return {
        "tokens": n,
        "attention_sink_first_token_mean": float(matrix[:, 0].mean()),
        "attention_sink_first_token_max": float(matrix[:, 0].max()),
        "causal_upper_triangle_max": float(upper.max()) if upper.size else 0.0,
        "causal_upper_triangle_sum": float(upper.sum()),
        "max_row_normalization_error": float(np.max(np.abs(rowsum - 1.0))),
        "late_query_attention_mass_by_key_position": thirds_mass,
    }


def reasoning_regions(tokens: list[str], prompt_length: int) -> dict[str, Any]:
    """通过实际 token 中的 <think> / </think> 边界定位，不伪造思考或答案区段。"""
    generated = list(range(prompt_length, len(tokens)))
    start = next((i for i in generated if "<think>" in tokens[i]), None)
    stop = next((i for i in generated if "</think>" in tokens[i]), None)
    if stop is not None:
        begin = start if start is not None else prompt_length
        thinking = list(range(begin, stop + 1))
        answer = list(range(stop + 1, len(tokens)))
    elif start is not None:
        thinking = list(range(start, len(tokens)))
        answer = []
    else:
        thinking = []
        answer = []  # 不凭空猜测普通输出为思考/最终回答
    answer_substantive = [i for i in answer if tokens[i].strip() and tokens[i] not in
                          ('<|im_end|>', '<|endoftext|>', '<|im_start|>')]
    return {
        "prompt_indices": list(range(prompt_length)),
        "thinking_indices": thinking,
        "answer_indices": answer,
        "answer_substantive_indices": answer_substantive,
        "explicit_think_open": start is not None,
        "explicit_think_close": stop is not None,
        "generated_tokens": len(generated),
    }


def region_attention_metrics(matrix, regions: dict) -> dict:
    import numpy as np

    answer = regions.get("answer_indices") or []
    if not answer:
        return {"available": False, "reason": "模型尚未输出 </think> 后的最终答案"}
    return {
        "available": True,
        "answer_query_attention_to": {
            label: float(matrix[np.ix_(answer, indexes)].sum(axis=1).mean()) if indexes else 0.0
            for label, indexes in [
                ("prompt", regions.get("prompt_indices") or []),
                ("thinking", regions.get("thinking_indices") or []),
                ("answer", regions.get("answer_indices") or []),
            ]
        },
    }


def draw_heatmaps(matrices: dict, tokens: list[str], target: Path, name: str,
                  boundary: int | None = None, answer_boundary: int | None = None):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # WSL/Linux 若装有中文字体就直接显示汉字；否则转换成 Unicode 转义，避免无字形警告。
    known = {item.name for item in font_manager.fontManager.ttflist}
    fonts = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    selected_font = next((font for font in fonts if font in known), None)
    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    total = len(matrices)
    fig, axes = plt.subplots(1, total, figsize=(5.9 * total, 5.2), squeeze=False)
    axes = axes[0]
    for ax, (layer, mat) in zip(axes, matrices.items()):
        shown = np.log10(np.maximum(mat, 1e-7))
        image = ax.imshow(shown, origin="upper", cmap="magma", aspect="auto", vmin=-7, vmax=0)
        ax.set_title(f"{layer}; sink={mat[:, 0].mean():.1%}")
        ax.set_xlabel("Key tokens")
        ax.set_ylabel("Query tokens")
        if boundary is not None and 0 < boundary < len(tokens):
            ax.axvline(boundary - .5, color="cyan", linestyle="--", linewidth=.9)
            ax.axhline(boundary - .5, color="cyan", linestyle="--", linewidth=.9)
        if answer_boundary is not None and 0 < answer_boundary < len(tokens):
            ax.axvline(answer_boundary - .5, color='lime', linestyle=':', linewidth=1.2,
                       label='Answer starts')
            ax.axhline(answer_boundary - .5, color='lime', linestyle=':', linewidth=1.2)
            ax.legend(loc='upper right', fontsize=7)
        if len(tokens) <= 26:
            tick_idx = list(range(len(tokens)))
            labels = [
                t.replace("\n", "\\n")[:9] if selected_font else t.encode("unicode_escape").decode("ascii")[:9]
                for t in tokens
            ]
            ax.set_xticks(tick_idx, labels, rotation=90, fontsize=7)
            ax.set_yticks(tick_idx, labels, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=.045, pad=.04, label="log10 attention")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(target, dpi=175, bbox_inches="tight")
    plt.close(fig)


def run(args, cfg: dict, root: Path) -> int:
    try:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("请安装：python -m pip install 'transformers>=4.51' torch numpy matplotlib tqdm pyyaml python-dotenv") from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可以改用 --device cpu，或检查 WSL 的 NVIDIA GPU 透传")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    run_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "experiment": "2-2", "created_at": datetime.now().isoformat(),
        "model": args.model, "device": device, "layers_requested": args.layers,
        "head": args.head, "simple_prompt": args.prompt, "reasoning_prompt": args.reasoning_prompt,
        "max_new_tokens": args.max_new_tokens, "host": platform.platform(),
        "protocol": {
            "canonical_model_id": 'Qwen/Qwen3-0.6B',
            "canonical_simple_prompt": '北京 的 天气 怎么样',
            "canonical_layers": [0, 13, -1],
            "canonical_reasoning_prompt": '请先认真推理，再用一句简洁的话回答：17 × 23 等于多少？',
            "canonical_max_new_tokens": 640,
            "canonical_seed": 20260730,
        },
        "note": '只以实际 Transformers eager attention 张量计算指标，绝不根据示意图补造数据。',
    }
    print(f"本地模型：{args.model}\n设备：{device}\n输出目录：{run_dir}")
    stage_count = 5 if args.max_new_tokens == 0 else 8
    try:
        with tqdm(total=stage_count, desc="实验2-2·总阶段", unit="阶段", dynamic_ncols=True) as pbar:
            pbar.set_description("2-2 加载 Tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            pbar.update(1)

            pbar.set_description("2-2 加载本地模型")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype="auto", attn_implementation="eager",
                local_files_only=True,
            ).to(device).eval()
            report["model_revision"] = getattr(model.config, "_commit_hash", None)
            report["model_type"] = getattr(model.config, "model_type", None)
            report["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
            report["local_model_identity"] = local_model_fingerprint(
                Path(args.model), include_weights=not args.skip_weight_hash)
            report["revision_note"] = (
                '本地文件不能仅凭目录名验证官方特定 commit。记录文件 SHA256，'
                '若 _commit_hash 不存在，则不能标记官方固定 revision 已核实。'
            )
            pbar.update(1)

            pbar.set_description("2-2 北京示例 Tokenize")
            simple_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            simple_id_list = simple_ids[0].tolist()
            simple_tokens = token_strings(tokenizer, simple_id_list)
            pbar.update(1)

            pbar.set_description("2-2 捕获原始注意力")
            # 先读 config，避免在 capture 之前假设有固定 28 层
            count = getattr(model.config, "num_hidden_layers", None)
            if count is None:
                raise RuntimeError("模型配置没有 num_hidden_layers；不是支持的 Transformer 因果语言模型")
            layers = resolve_layers(args.layers, int(count))
            simple_mats, num_layers, num_heads = capture_attention(model, simple_ids, layers, args.head)
            pbar.update(1)

            pbar.set_description("2-2 保存简单热力图")
            arrays = {"simple_" + name: value for name, value in simple_mats.items()}
            report["simple"] = {
                "token_ids": simple_id_list, "tokens": simple_tokens,
                "metrics": {name: layer_metrics(mat) for name, mat in simple_mats.items()},
            }
            draw_heatmaps(simple_mats, simple_tokens, run_dir / "beijing_attention.png", "Experiment 2-2: Input attention")
            pbar.update(1)

            report["architecture"] = {"layers": num_layers, "attention_heads": num_heads}
            if args.max_new_tokens:
                pbar.set_description("2-2 生成实际推理序列")
                messages = [
                    {"role": "system", "content": "你是能认真推理并给出准确答案的助手。"},
                    {"role": "user", "content": args.reasoning_prompt},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                input_ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                context_n = int(input_ids.shape[1])
                torch.manual_seed(args.seed)
                # 每生成一个 token 更新一次 tqdm，而不是只能等整段生成结束。
                from transformers import StoppingCriteria, StoppingCriteriaList

                class TokenProgress(StoppingCriteria):
                    def __init__(self, initial_length, bar):
                        self.previous = initial_length
                        self.bar = bar

                    def __call__(self, generated_ids, scores, **kwargs):
                        length = int(generated_ids.shape[-1])
                        if length > self.previous:
                            self.bar.update(length - self.previous)
                            self.previous = length
                        return False  # 只记录进度，不改变模型停止条件

                with tqdm(total=args.max_new_tokens, desc="2-2 真实 Token 生成", unit="token", dynamic_ncols=True) as genbar:
                    with torch.inference_mode():
                        full_ids = model.generate(
                            input_ids=input_ids, max_new_tokens=args.max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([TokenProgress(context_n, genbar)]),
                        )
                    actually_generated = int(full_ids.shape[1]) - context_n
                    if actually_generated > genbar.n:
                        genbar.update(actually_generated - genbar.n)
                pbar.update(1)

                pbar.set_description("2-2 捕获生成后注意力")
                ids = full_ids[0].tolist()
                tokens = token_strings(tokenizer, ids)
                generation_mats, _, _ = capture_attention(model, full_ids, layers, args.head)
                arrays.update({"generated_" + name: mat for name, mat in generation_mats.items()})
                regions = reasoning_regions(tokens, context_n)
                report["generated"] = {
                    "context_tokens": context_n,
                    "rendered_chat_template": rendered,
                    "all_token_ids": ids,
                    "generated_token_ids": ids[context_n:],
                    "generated_text": tokenizer.decode(ids[context_n:], skip_special_tokens=False),
                    "all_tokens": tokens,
                    "regions": regions,
                    "metrics": {name: layer_metrics(mat) for name, mat in generation_mats.items()},
                    "region_attention": {name: region_attention_metrics(mat, regions) for name, mat in generation_mats.items()},
                }
                pbar.update(1)

                pbar.set_description("2-2 保存推理热力图")
                draw_heatmaps(
                    generation_mats, tokens, run_dir / 'reasoning_attention.png',
                    'Experiment 2-2: Prompt + reasoning + answer', boundary=context_n,
                    answer_boundary=(regions['answer_indices'][0] if regions['answer_indices'] else None),
                )
                pbar.update(1)

        np.savez_compressed(run_dir / "attention_matrices.npz", **arrays)
        simple_metrics = report.get('simple', {}).get('metrics', {})
        generated_metrics = report.get('generated', {}).get('metrics', {})
        masks_ok = all(layer_metrics(mat)['causal_upper_triangle_max'] <= 1e-7
                       for mat in arrays.values())
        normalizations_ok = all(layer_metrics(mat)['max_row_normalization_error'] <= 1e-3
                                for mat in arrays.values())
        region = report.get('generated', {}).get('regions', {})
        resolved_expected_layers = resolve_layers('0,13,-1', report['architecture']['layers'])
        canonical_run = (
            args.prompt == '北京 的 天气 怎么样' and args.max_new_tokens == 640 and
            args.reasoning_prompt == '请先认真推理，再用一句简洁的话回答：17 × 23 等于多少？' and
            args.seed == 20260730 and layers == resolved_expected_layers and args.head == -1
        )
        checks = {
            'real_qwen3_architecture': report.get('model_type') == 'qwen3',
            'local_weights_fingerprinted': bool(report.get('local_model_identity', {}).get('files')),
            'three_layers_captured': len(simple_mats) == 3 and
                                     (args.max_new_tokens == 0 or len(generation_mats) == 3),
            'lossless_attention_matrices_saved': (run_dir / 'attention_matrices.npz').stat().st_size > 0,
            'simple_heatmap_saved': (run_dir / 'beijing_attention.png').stat().st_size > 0,
            'generated_heatmap_saved': (run_dir / 'reasoning_attention.png').stat().st_size > 0
                                       if args.max_new_tokens else False,
            'causal_triangle_verified': masks_ok,
            'attention_rows_normalized': normalizations_ok,
            'explicit_think_open': bool(region.get('explicit_think_open')),
            'explicit_think_close': bool(region.get('explicit_think_close')),
            'final_answer_content_present': bool(region.get('answer_substantive_indices')),
            'reasoning_region_metrics_saved': bool(report.get('generated', {}).get('region_attention')),
            'position_mass_measured': bool(simple_metrics) and all(
                bool(stats.get('late_query_attention_mass_by_key_position'))
                for stats in simple_metrics.values()),
        }
        report['checks'] = checks
        report['status'] = 'complete' if all(checks.values()) else 'partial'
        report['canonical_default_settings'] = canonical_run
        report['pinned_commit_available'] = (
            isinstance(report.get('model_revision'), str) and len(report['model_revision']) == 40)
        report['official_exact_revision_verified'] = (
            report['pinned_commit_available'] and
            bool(args.expected_revision) and report['model_revision'] == args.expected_revision)
        report['official_exact_complete'] = (
            report['status'] == 'complete' and canonical_run and
            report['official_exact_revision_verified'])
        if not args.max_new_tokens:
            report['note'] = '只生成北京输入热力图：属于局部 smoke，思考和最终回答未测，不是完整实验。'
        elif not region.get('answer_substantive_indices'):
            report['note'] = '实际生成未出现完整的 </think> 后答案；增大 --max-new-tokens，结果标为 partial。'
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=6)
        print(f"[运行失败] {report['error']}", file=sys.stderr)
    finally:
        report["finished_at"] = datetime.now().isoformat()
        save_json(run_dir / "experiment_results.json", report)
        manifest = {
            "experiment": "2-2", "status": report.get("status"),
            "checks": report.get('checks', {}),
            "canonical_default_settings": report.get('canonical_default_settings', False),
            "official_exact_complete": report.get('official_exact_complete', False),
            "files": {file.name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)} for file in run_dir.iterdir() if file.is_file()},
        }
        save_json(run_dir / "manifest.json", manifest)
        print(f"结果已保存：{run_dir}")
    if report.get("simple"):
        print("\n[注意力统计]")
        for name, stats in report["simple"]["metrics"].items():
            print(f"  {name} | Sink平均占比 {stats['attention_sink_first_token_mean']:.1%} | 因果上三角最大值 {stats['causal_upper_triangle_max']:.2e}")
    if report.get("generated"):
        print(f"[思考区] {len(report['generated']['regions']['thinking_indices'])} token | [答案区] {len(report['generated']['regions']['answer_indices'])} token")
    return 0 if report.get('status') == 'complete' or (args.max_new_tokens == 0 and report.get('status') == 'partial') else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="实验 2-2：读取本地 Hugging Face Qwen 模型的真实 Attention 权重并绘图")
    parser.add_argument("--model", help="WSL 本地模型目录（默认 /mnt/e/ai-agent-book/qwen3_0.6b），不能填写 Hugging Face 远程模型名")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    parser.add_argument("--prompt", default="北京 的 天气 怎么样")
    parser.add_argument("--reasoning-prompt", default="请先认真推理，再用一句简洁的话回答：17 × 23 等于多少？")
    parser.add_argument("--layers", default="0,13,-1", help="以逗号分隔的层索引，比如 0,13,-1")
    parser.add_argument("--head", type=int, default=-1, help="-1 表示对所有注意力头求平均")
    parser.add_argument("--max-new-tokens", type=int, default=640, help="生成多少 token（0 只生成北京热力图）")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-dir", help="结果目录，默认本脚本所在目录的 results 子目录")
    parser.add_argument("--local-files-only", action="store_true", help="兼容旧参数；本版本默认强制仅从本地加载")
    parser.add_argument("--skip-weight-hash", action="store_true", help="跳过权重 SHA256，仅记录配置文件身份；不影响注意力实验")
    parser.add_argument("--expected-revision", help="可选：已独立核实的官方 HF commit SHA，用于严格身份验收；不要猜测")
    parser.add_argument("--dry-run", action="store_true", help="打印配置，不下载模型和运行推理")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens 不能为负数")
    if args.head < -1:
        parser.error("--head 只能为 -1 或非负整数")

    root = project_root()
    cfg = read_config(root)
    args.model = args.model or cfg.get("local_model_dir") or cfg.get("model") or str(DEFAULT_LOCAL_MODEL_DIR)
    args.local_files_only = True
    args.device = args.device or cfg.get("device", "auto")
    # 默认结果目录跟随脚本位置；自定义相对路径则相对项目根目录。
    args.output_dir = args.output_dir or cfg.get("output_dir") or str(Path(__file__).resolve().parent / "results")
    if args.dry_run:
        path = Path(args.model).expanduser()
        print(json.dumps({
            "experiment": "2-2", "model": args.model, "device": args.device,
            "model_dir_exists": path.is_dir(),
            "config_exists": (path / "config.json").is_file(),
            "weights_found": any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")),
            "tokenizer_exists": any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")),
            "local_files_only": True,
            "simple_prompt": args.prompt, "reasoning_prompt": args.reasoning_prompt,
            "layers": args.layers, "head": args.head, "max_new_tokens": args.max_new_tokens,
            "skip_weight_hash": args.skip_weight_hash,
            "expected_revision": args.expected_revision,
            "output_dir": args.output_dir,
            "note": "真实注意力必须使用本地 Hugging Face 模型；百炼聊天 API 不返回 Attention 张量。",
        }, ensure_ascii=False, indent=2))
        return 0
    try:
        args.model = str(validate_local_model_dir(args.model))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return run(args, cfg, root)


if __name__ == "__main__":
    raise SystemExit(main())
