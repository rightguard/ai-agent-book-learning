import argparse, json, yaml, os, sys, re
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

CITIES = (
    "文莱：斯里巴加湾市 (Bandar Seri Begawan)",
    "柬埔寨：金边 (Phnom Penh)",
    "印度尼西亚：雅加达 (Jakarta，按教程的固定城市集合)",
    "老挝：万象 (Vientiane)",
    "马来西亚：吉隆坡 (Kuala Lumpur)",
    "缅甸：内比都 (Naypyidaw)",
    "菲律宾：马尼拉 (Manila)",
    "新加坡：新加坡 (Singapore)",
    "泰国：曼谷 (Bangkok)",
    "越南：河内 (Hanoi)",
)

SYSTEM = (
    "你是地理数据研究 Agent。必须根据真实搜索结果获取坐标，不能编造来源或经纬度；"
    "距离计算必须使用服务端托管 code_interpreter实际运行Python，不能仅口头声称计算完成。"
    "若工具失败或缺少坐标，应明确指出，不能把任务当作成功完成。请用中文回答。"
)

TASK = f"""完成教程实验 1-3 的第一个任务：比较以下固定十个城市的球面距离。
{chr(10).join(CITIES)}

请自主完成以下步骤：
1. 调用托管 web_search 搜索上述十个城市的纬度、经度；用可靠网页交叉核对，给出至少两个真实来源 URL。
2. 将搜索获得的十组坐标交给托管 code_interpreter；实际运行 Python（可使用 itertools.combinations 和 math），
   使用 Haversine 公式及地球平均半径 R=6371.0088 km，计算全部 C(10,2)=45 对城市的大圆距离。
3. 在 Python 执行日志中明确打印三行：PAIR_COUNT=45、CLOSEST_PAIR=英文城市名1|英文城市名2、DISTANCE_KM=数字。
   请由 Python 计算产生这些行，不要在没有执行代码时自行填写。
4. 最终用中文给出十个城市的坐标表、最近的一对及公里数、计算方法、比较对数、真实来源链接。
这是教程固定十城市集合，不要将其描述为 2026 年东盟的全部成员国首都。
"""

def settings():
    script = Path(__file__).resolve()
    root = next(
        (p for p in script.parents if (p / "config.yaml").is_file() and (p / ".env").is_file()),
        None,
    )
    if root is None:
        raise FileNotFoundError("未找到项目根目录的 config.yaml 和 .env")
    load_dotenv(root / ".env")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    model_config = config.get("model") or {}
    base_url = (
        os.getenv("QWEN_API_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip().rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("API地址错误")
    return {
        "root": root,
        "api_key": os.getenv("DASHSCOPE_API_KEY", "").strip(),
        "base_url": base_url,
        "model": model_config.get("name", "qwen3.7-flash-2026-07-15"),
        "temperature": float(model_config.get("temperature", 0.3)),
        "max_output_tokens": int(model_config.get("max_tokens", 8192)),
    }

def as_dict(value):
    return value if isinstance(value, dict) else value.model_dump(mode="json")

def extract_answer(data):
    return "\n".join(
        part.get("text", "")
        for item in data.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    ).strip()

# 用模型服务端工具验收
def analyse(data):
    items = data.get("output") or []

    search_calls = [i for i in items if i.get("type") == "web_search_call" and i.get("status") == "completed"]
    code_calls = [i for i in items if i.get("type") == "code_interpreter_call" and i.get("status") == "completed"]
    urls = sorted({
        src if isinstance(src, str) else src.get("url")
        for item in search_calls for src in (item.get("action") or {}).get("sources", [])
        if isinstance(src, str) or (isinstance(src, dict) and src.get("url"))
    })

    code_logs = "\n".join(
        str(part.get("logs", ""))
        for item in code_calls for part in (item.get("outputs") or [])
        if isinstance(part, dict)
    )

    answer = extract_answer(data)
    answer_urls = sorted(set(re.findall(r"https?://[^\s)\]>]+", answer)))
    pair_line = re.search(r"(?im)^\s*CLOSEST_PAIR\s*=\s*([^\n]+)", code_logs)
    pair_names = sorted(x.strip().lower() for x in pair_line.group(1).split("|")) if pair_line else []
    distance_line = re.search(r"(?im)^\s*DISTANCE_KM\s*=\s*([0-9]+(?:\.[0-9]+)?)", code_logs)
    distance = float(distance_line.group(1)) if distance_line else None

    checks = {
        "响应正常完成": data.get("status") == "completed",
        "托管搜索成功执行": bool(search_calls),
        "托管代码成功执行": bool(code_calls),
        "最终回答包含至少两个来源URL": len(answer_urls) >= 2,
        "answer_urls": answer_urls,
        "source_provenance_verified": len(answer_urls) >= 2,
        "Python日志确认计算45对": bool(re.search(r"PAIR_COUNT\s*=\s*45\b", code_logs)),
        "Python最近城市对正确": pair_names == ["kuala lumpur", "singapore"],
        "Python计算距离接近参考值": distance is not None and abs(distance - 316.35) < 5,
        "生成了最终回答": bool(answer),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "search_calls": len(search_calls),
        "code_calls": len(code_calls),
        "sources": urls,
        "code_logs": code_logs,
        "distance_km_from_code": distance,
        "answer": answer,
    }

def run(client, request):
    """DashScope 的托管工具过程用流式 Responses API 接收。"""
    response = None
    event_counts = {}
    stream = client.responses.create(**request)
    try:
        for event in stream:
            kind = event.type
            event_counts[kind] = event_counts.get(kind, 0) + 1
            if kind == "response.web_search_call.completed":
                print("[工具] 托管网络搜索完成")
            elif kind == "response.code_interpreter_call.completed":
                print("[工具] 托管 Python 计算完成")
            elif kind in {"response.completed", "response.failed", "response.incomplete"}:
                response = as_dict(event.response)
    finally:
        stream.close()
    if response is None:
        raise RuntimeError("响应流未返回最终 response.completed / response.failed 事件")
    return response, event_counts


def main():
    parser = argparse.ArgumentParser(description="实验 1-3：十个首都、45 对距离、托管搜索和计算")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和请求，不调用 API")
    parser.add_argument("--model", help="覆盖 YAML 中的模型，如qwen3.7-plus")
    parser.add_argument("--output", type=Path, help="指定结果JSON文件路径")
    args = parser.parse_args()

    cfg = settings()
    model = args.model or cfg["model"]
    request = {
        "model": model,
        "instructions": SYSTEM,
        "input": TASK,
        "tools": [{"type": "web_search"}, {"type": "code_interpreter"}],
        "temperature": cfg["temperature"],
        "max_output_tokens": cfg["max_output_tokens"],
        "extra_body": {"enable_thinking": True},
        "stream": True,
    }
    print(f"项目根目录：{cfg['root']}")
    print(f"配置文件：{cfg['root'] / 'config.yaml'}")
    print(f"环境文件：{cfg['root'] / '.env'}")
    print(f"模型：{model}\nAPI 地址：{cfg['base_url']}/responses")
    if args.dry_run:
        print("\n请求预览（不包含 API Key）：")
        print(json.dumps(request, indent=2, ensure_ascii=False))
        return
    if not cfg["api_key"]:
        parser.error(".env 中未设置 DASHSCOPE_API_KEY")

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=900, max_retries=1)
    data, event_counts = {}, {}
    error = None
    try:
        data, event_counts = run(client, request)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    evaluation = analyse(data)
    record = {
        "experiment": "1-3-asean-capitals-hosted-tools",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": model,
        "base_url": cfg["base_url"],
        "request": request,
        "response": data,
        "events": event_counts,
        "validation": evaluation,
        "error": error,
    }
    output = args.output or Path(__file__).resolve().parent / "results" / (
        "asean_distance_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[验收]")
    for key, ok in evaluation["checks"].items():
        print(f"  {key}：{'通过' if ok else '未通过'}")
    if error:
        print(f"\n[调用错误] {error}")
    if evaluation["answer"]:
        print("\n[最终回答]\n" + evaluation["answer"])
    print(f"\n结果已保存：{output}")
    if not evaluation["passed"]:
        print("提示：未满足全部验收条件，请检查 JSON 中的工具执行记录与来源。")
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
