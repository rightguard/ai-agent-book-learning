import argparse, base64, hashlib, json, os, re, sys, datetime, requests, yaml, tqdm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


REWRITE_SYSTEM = (
    "你是文生图提示词改写助手。请把用户的中文需求改写成适合图像生成模型使用的高质量中文提示词。"
    "要求：1）保留原始需求中的核心约束、主体、文案、氛围与风格；"
    "2）补充必要但不过度的构图、光线、材质、镜头、背景细节；"
    "3）不要擅自删除用户要求的文字文案；"
    "4）只输出JSON，对象字段为prompt和notes；"
    "5）prompt用中文完整描述，不要输出markdown。"
)


def find_root() -> Path:
    file_path = Path(__file__).resolve()
    root = next(
        (
            p
            for p in [file_path.parent, *file_path.parents]
            if (p / "config.yaml").is_file() and (p / ".env").is_file()
        ),
        None,
    )
    if root is None:
        raise FileNotFoundError("根目录错误")
    return root
# .hexdigest()把哈希结果转换成 64 位十六进制字符串，方便保存和比较
def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_settings():
    root = find_root()
    load_dotenv(root / ".env")

    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    model_cfg = cfg.get("model") or {}
    image_cfg = cfg.get("image") or {}
    exp_cfg = cfg.get("experiment_1_4") or {}

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("未找到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    base_url = (
        os.environ.get("QWEN_API_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    settings = {
        "root": root,
        "api_key": api_key,
        "base_url": base_url,
        "image_base_url": os.getenv("QWEN_IMAGE_API_BASE_URL") or base_url,
        "text_model": model_cfg.get("name", "qwen3.7-flash-2026-07-15"),
        "temperature": float(model_cfg.get("temperature", 0.3)),
        "top_p": float(model_cfg.get("top_p", 0.9)),
        "max_tokens": int(model_cfg.get("max_tokens", 8192)),
        "image_model": image_cfg.get("model", "qwen-image-3.0-pro"),
        "image_size": image_cfg.get("size", "1024x1024"),
        "image_n": int(image_cfg.get("n", 1)),
        "image_seed": image_cfg.get("seed", 42),
        "image_timeout": int(image_cfg.get("timeout", 600)),
        "prompt_extend": bool(image_cfg.get("prompt_extend", False)),
        "watermark": bool(image_cfg.get("watermark", False)),
        "routes": list(exp_cfg.get("routes") or ["direct", "rewritten"]),
        "output_dir": exp_cfg.get("output_dir", "char1/image-gen-workflow/results"),
        "cases": list(exp_cfg.get("cases") or []),
    }

    if not settings["cases"]:
        raise ValueError("config.yaml 中 experiment_1_4.cases 为空，请先补充实验用例")

    return settings

def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

# 这个函数相当于给 LLM 的 JSON 输出增加了一层容错处理。先正常解析，失败时尝试提取JSON，再次解析；但它无法修复所有格式错误，也不保证提取到的内容一定有效
def safe_json_loads(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))

def rewrite_prompt(client: OpenAI, cfg: Dict, prompt: str) -> Dict:
    resp = client.chat.completions.create(
        model=cfg["text_model"],
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=min(cfg["max_tokens"], 8192),
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    content = resp.choices[0].message.content or "{}"
    obj = safe_json_loads(content)

    return {
        "raw_response": content,
        "rewritten_prompt": str(obj.get("prompt") or "").strip() or prompt,
        "notes": str(obj.get("notes") or "").strip(),
        "usage": (resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else None) if getattr(resp, "usage", None) else None,
    }

def download_url(url:str, timeout: int=600) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

# 无论图像生成 API 返回的是 Base64 编码的图片，还是图片下载 URL，都统一转换成 bytes（图片的二进制数据），方便后续保存到本地。
def image_result_to_bytes(item, timeout: int) -> bytes:
    # 例如，一张图片实际上由大量二进制字节构成。但有些 API 使用 JSON 返回数据
    # 不能直接将原始二进制字节放进标准 JSON 字符串中，所以会采用 Base64 编码。
    if hasattr(item, "b64_json") and item.b64_json:
        return base64.b64decode(item.b64_json)
    if isinstance(item, dict) and item.get("b64_json"):
        return base64.b64decode(item["b64_json"])

    url = getattr(item, "url", None) if not isinstance(item, dict) else item.get("url")
    if url:
        return download_url(url, timeout=timeout)

    # 一些sdk可能序列化为字典
    if hasattr(item, "model_dump"):
        # model_dump() 的主要作用是把 Pydantic 模型对象转换成 Python 字典
        dumped = item.model_dump()
        if dumped.get("b64_json"):
            return base64.b64decode(dumped["b64_json"])
        if dumped.get("url"):
            return download_url(dumped["url"], timeout=timeout)

    raise ValueError(f"图像响应中未找到b64_json或url")

def generate_image(client: OpenAI, cfg: Dict, prompt: str):
    # extra_body 是一个普通 Python 字典，用来传递图像模型服务商特有的参数
    # prompt_extend=False：请求关闭模型内部的自动提示词扩写，便于控制实验变量。
    # watermark=False：请求不添加水印。
    extra_body = {
        "prompt_extend": cfg["prompt_extend"],
        "watermark": cfg["watermark"],
    }
    if cfg["image_seed"] is not None:
        extra_body["seed"] = cfg["image_seed"]

    image_client = client.with_options(
        base_url=cfg["image_base_url"],
        timeout=cfg["image_timeout"],
    )

    resp = image_client.images.generate(
        model=cfg["image_model"],
        prompt=prompt,
        size=cfg["image_size"],
        n=cfg["image_n"],
        extra_body=extra_body,
    )

    data = getattr(resp, "data", None) or []
    if not data:
        raise ValueError("图像生成接口未返回任何data")

    # 将返回结果转换成图片字节
    content = image_result_to_bytes(data[0], timeout=cfg["image_timeout"])
    usage = None # 本次图像生成的用量信息

    if hasattr(resp, "usage") and resp.usage is not None:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else resp.usage
    return content, resp, usage

def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def route_label(route: str) -> str:
    return {"direct": "原始需求直出的图",
            "rewritten": "改写后生成的图"}.get(route, route)

def run_one_case(client: OpenAI, cfg: Dict, case: Dict, route: str, run_dir: Path, seq: int):
    original_prompt = str(case["prompt"])
    rewrite_info = None
    final_prompt = original_prompt

    if route == "rewritten":
        rewrite_info = rewrite_prompt(client=client, cfg=cfg, prompt=original_prompt)
        final_prompt = rewrite_info["rewritten_prompt"]

    image_bytes, img_resp, image_usage = generate_image(client=client, cfg=cfg, prompt=final_prompt)

    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{seq:02d}_{case['id']}_{route}.png"
    img_path = image_dir / img_name
    img_path.write_bytes(image_bytes)

    result = {
        "case_id": case["id"],
        "category": case.get("category", "unknown"),
        "route": route,
        "route_label": route_label(route),
        "original_prompt": original_prompt,
        "final_prompt": final_prompt,
        "rewrite": rewrite_info,
        "image_file": str(img_path.relative_to(run_dir)),
        "image_sha256": sha256_bytes(image_bytes),
        "image_size_bytes": len(image_bytes),
        "image_request": {
            "model": cfg["image_model"],
            "size": cfg["image_size"],
            "n": cfg["image_n"],
            "prompt_extend": cfg["prompt_extend"],
            "seed": cfg["image_seed"],
            "watermark": cfg["watermark"],
        },
        "image_usage": image_usage,
        "image_response": {
            "request_id": getattr(img_resp, "_request_id", None),
            "created": getattr(img_resp, "created", None),
            "data_count": len(getattr(img_resp, "data", None) or []),
},
    }
    return result

def build_tasks(cfg: Dict, only_case: Optional[str], only_route: Optional[str]) -> List[Dict]:
    routes = cfg["routes"]
    if only_route:
        if only_route not in routes:
            raise ValueError(f"指定的 route 不在配置中: {only_route}")
        routes = [only_route]

    cases = cfg["cases"]
    if only_case:
        cases = [c for c in cases if c.get("id") == only_case]
        if not cases:
            raise ValueError(f"未找到指定的 case: {only_case}")

    tasks = []
    for case in cases:
        for route in routes:
            tasks.append({"case": case, "route": route})
    return tasks


def main():
    parser = argparse.ArgumentParser(description="实验 1-4：提示词改写与原生图像生成对照")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不发起 API 请求")
    parser.add_argument("--case", dest="only_case", help="只运行指定 case id")
    parser.add_argument("--route", dest="only_route", choices=["direct", "rewritten"], help="只运行指定路线")
    parser.add_argument("--image-model", help="临时覆盖图像模型")
    parser.add_argument("--text-model", help="临时覆盖提示词改写模型")
    args = parser.parse_args()

    cfg = load_settings()
    image_base_url = cfg["image_base_url"].rstrip("/")
    print("图像API请求地址：", image_base_url + "/images/generations")

    if not image_base_url.endswith("/compatible-mode/v1"):
        raise ValueError("图像API地址配置错误，请检查 QWEN_IMAGE_API_BASE_URL")
    if args.image_model:
        cfg["image_model"] = args.image_model
    if args.text_model:
        cfg["text_model"] = args.text_model

    tasks = build_tasks(cfg, args.only_case, args.only_route)

    print(f"项目根目录：{cfg['root']}")
    print(f"配置文件：{cfg['root'] / 'config.yaml'}")
    print(f"环境文件：{cfg['root'] / '.env'}")
    print(f"文本模型：{cfg['text_model']}")
    print(f"图像模型：{cfg['image_model']}")
    print(f"输出目录：{cfg['root'] / cfg['output_dir']}")
    print(f"实验任务数：{len(tasks)}")

    if args.dry_run:
        print("\n[待运行任务]")
        for i, task in enumerate(tasks, start=1):
            case = task["case"]
            print(f"  {i}. {case['id']} | {case.get('category', 'unknown')} | {task['route']} | {case['prompt']}")
        return

    client = make_client(cfg["api_key"], cfg["base_url"])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg["root"] / cfg["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "1-4-image-workflow-qwen",
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "text_model": cfg["text_model"],
        "image_model": cfg["image_model"],
        "image_size": cfg["image_size"],
        "prompt_extend": cfg["prompt_extend"],
        "tasks_total": len(tasks),
        "tasks": [],
        "failures": [],
    }

    pbar = tqdm(total=len(tasks), desc="实验1-4·图片生成总进度", ncols=100)
    for seq, task in enumerate(tasks, start=1):
        case = task["case"]
        route = task["route"]
        pbar.set_postfix_str(f"{case['id']} | {route}")
        try:
            result = run_one_case(client, cfg, case, route, run_dir, seq)
            manifest["tasks"].append(result)
            print(f"\n[完成] {case['id']} | {route_label(route)} -> {result['image_file']}")
        except Exception as exc:
            # 打印 API 返回的详细错误，方便排查 404
            response = getattr(exc, "response", None)

            if response is not None:
                print(f"\n请求地址：{response.request.url}")
                print(f"HTTP状态码：{response.status_code}")
                print(f"错误详情：{response.text[:2000]}")
            err = {
                "case_id": case.get("id"),
                "route": route,
                "error": f"{type(exc).__name__}: {exc}",
            }
            manifest["failures"].append(err)
            print(f"\n[失败] {case.get('id')} | {route_label(route)} | {err['error']}", file=sys.stderr)

            if response is not None and response.status_code in (401, 403, 404):
                print("检测到API接口或权限错误，停止后续实验。", file=sys.stderr)
                break
        finally:
            pbar.update(1)
    pbar.close()

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["success_count"] = len(manifest["tasks"])
    manifest["failure_count"] = len(manifest["failures"])
    manifest["passed"] = manifest["success_count"] == len(tasks)

    save_json(run_dir / "manifest.json", manifest)

    # 生成简单对照索引
    summary = {
        "run_id": run_id,
        "passed": manifest["passed"],
        "success_count": manifest["success_count"],
        "failure_count": manifest["failure_count"],
        "images": [
            {
                "case_id": x["case_id"],
                "route": x["route"],
                "image_file": x["image_file"],
                "original_prompt": x["original_prompt"],
                "final_prompt": x["final_prompt"],
            }
            for x in manifest["tasks"]
        ],
    }
    save_json(run_dir / "summary.json", summary)

    print("\n[结果汇总]")
    print(f"  成功：{manifest['success_count']} / {len(tasks)}")
    print(f"  失败：{manifest['failure_count']}")
    print(f"  结果目录：{run_dir}")
    print(f"  Manifest：{run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

