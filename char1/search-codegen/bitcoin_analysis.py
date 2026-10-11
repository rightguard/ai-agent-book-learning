import argparse, json, os, re, yaml
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm


# 故意保持需求模糊，观察agent是否先询问，而不是直接调用工具。
INITIAL_TASK = "搜索最近一个月的比特币走势，做技术分析。"


DEFAULT_REPLY = (
    f"查询基准日期（UTC）：{datetime.now(timezone.utc).date().isoformat()}。"
    "使用 CoinGecko 作为首选数据源，分析 BTC/USD 最近 30 个完整 UTC 日的走势。"
    "需要 MA7、MA20、RSI14、MACD(12,26,9)、区间收益率、最大回撤，并尽可能绘制收盘价走势图。"
    "请搜索可追溯的真实日线收盘价，并通过托管 Python 实际计算。"
    "为了计算 MACD，尽量取得至少 60 天（最好 90 天）连续有效日线，"
    "只展示最近 30 天的分析结果。"
    "如果 CoinGecko 数据不可访问或日线不完整，请说明限制，不要编造数据、日期或来源。"
)

SYSTEM = (
    "你是数字资产数据研究agent，请使用中文回答。"
    "硬性规则：当用户首次提出含糊的研究需求，没有明确数据源、币种和技术指标时，"
    "必须先用简短的问题澄清至少【数据源】和【指标】；在获得用户补充前，禁止调用任何工具。"
    "用户回答明确后，使用服务端托管 web_search 搜集可信的真实数据和来源；"
    "需要时使用 web_extractor 阅读网页；必须使用服务端托管 code_interpreter "
    "实际运行 Python 完成所有数值计算，不能口算或声称运行了代码。"
    "日期统一按 UTC 说明，并写明数据区间、收盘价口径、来源和数据不足之处。"
    "不同来源的价格、采样频率、交易日定义不能混用；不可编造缺失价格。"
    "若可获得足够的实际日线数据，请计算 MA7、MA20、RSI14、"
    "MACD(12,26,9)及信号线、最近30日区间收益率和最大回撤。"
    "请让 Python 运行日志分别打印如下以便程序自动验收："
    "DATA_POINTS=数字、MA7=数字、MA20=数字、RSI14=数字、"
    "MACD=数字、MACD_SIGNAL=数字、PERIOD_RETURN_PCT=数字、"
    "MAX_DRAWDOWN_PCT=数字。"
    "这些值必须来自实际代码执行；无法计算时明确说明，绝不能凭空补齐这些行。"
    "技术指标只做教学分析，不提供确定收益保证。"
)


def settings():
    script = Path(__file__).resolve()
    root = next(
        (p for p in script.parents
         if (p / "config.yaml").is_file() and (p / ".env").is_file()),
        None,
    )
    if root is None:
        raise FileNotFoundError("未找到项目根目录")

    load_dotenv(root / ".env")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8-sig")) or {}
    model_config = config.get("model") or {}

    base_url = (
        os.getenv("QWEN_API_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip().rstrip("/")

    if not base_url.endswith("/v1"):
        raise ValueError("API 根地址必须以 /compatible-mode/v1 结尾；不要附加 /responses")

    return {
        "root": root,
        "api_key": os.getenv("DASHSCOPE_API_KEY", "").strip(),
        "base_url": base_url,
        "model": str(model_config.get("name", "qwen3.7-flash-2026-07-15")),
        "temperature": float(model_config.get("temperature", 0.3)),
        "max_output_tokens": int(model_config.get("max_tokens", 8192)),
    }


def as_dict(value):
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")

#两轮都声明工具，由模型决定是否调用
def build_request(cfg, prompt, previous_response_id=None, with_extractor=False):
    tools = [{"type": "web_search"}, {"type": "code_interpreter"}]
    if with_extractor:
        tools.append({"type": "web_extractor"})

    request = {
        "model": cfg["model"],
        "instructions": SYSTEM,
        "input": prompt,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": cfg["temperature"],
        "max_output_tokens": cfg["max_output_tokens"],
        "extra_body": {"enable_thinking": True},
        "store": True,  # 支持第二轮使用previous_response_id关联上下文
        "stream": True,
    }

    if previous_response_id:
        request["previous_response_id"] = previous_response_id
    return request

# 只提取模型回答，不把推理和工具日志当作答案
def extract_answer(data):
    chunks = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                chunks.append(part.get("text") or "")
    return "\n".join(chunks).strip()


def calls_of(data, name):
    return [
        item for item in data.get("output") or []
        if isinstance(item, dict)
        and item.get("type") == name
        and item.get("status") == "completed"
    ]


def all_tool_items(data):
    names = {"web_search_call", "web_extractor_call", "code_interpreter_call"}
    return [
        item for item in data.get("output") or []
        if isinstance(item, dict) and item.get("type") in names
    ]


def collect_sources(data):
    urls = set()
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            for source in action.get("sources") or []:
                url = source if isinstance(source, str) else (
                    source.get("url") if isinstance(source, dict) else None
                )
                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    urls.add(url)
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            for annotation in part.get("annotations") or []:
                if isinstance(annotation, dict):
                    url = annotation.get("url")
                    if isinstance(url, str) and url.startswith(("https://", "http://")):
                        urls.add(url)
    return sorted(urls)


def extract_code_logs(data):
    logs, codes = [], []
    for item in calls_of(data, "code_interpreter_call"):
        if item.get("code"):
            codes.append(str(item["code"]))
        for part in item.get("outputs") or []:
            if isinstance(part, dict):
                message = part.get("logs") or part.get("text")
                if message:
                    logs.append(str(message))
    return "\n".join(logs), codes


def read_marker(logs, name):
    pattern = rf"(?mi)^\s*{re.escape(name)}\s*=\s*([-+]?\d+(?:\.\d+)?)\b"
    match = re.search(pattern, logs)
    return float(match.group(1)) if match else None


def analyze(first, second, second_request=None):
    answer1 = extract_answer(first)
    answer2 = extract_answer(second)
    first_tool_items = all_tool_items(first)
    searches = calls_of(second, "web_search_call")
    codes = calls_of(second, "code_interpreter_call")
    extractors = calls_of(second, "web_extractor_call")
    logs, executed_code = extract_code_logs(second)

    markers = {
        name: read_marker(logs, name)
        for name in (
            "DATA_POINTS", "MA7", "MA20", "RSI14", "MACD",
            "MACD_SIGNAL", "PERIOD_RETURN_PCT", "MAX_DRAWDOWN_PCT"
        )
    }
    source_urls = collect_sources(second)
    answer_urls = sorted(set(
        re.findall(r"https?://[^\s)\]>\"']+", answer2)
    ))
    is_question = ("？" in answer1 or "?" in answer1)
    first_clarified = (
        first.get("status") == "completed"
        and bool(answer1) and is_question and not first_tool_items
    )

    checks = {
        "第一轮正常完成": first.get("status") == "completed",
        "第一轮先澄清且没有调用工具": first_clarified,
        "第二轮通过previous_response_id续接": bool(
            second_request and first.get("id")
            and second_request.get("previous_response_id") == first["id"]
        ),
        "第二轮正常完成": second.get("status") == "completed",
        "托管搜索真实完成": bool(searches),
        "托管Python真实完成": bool(codes),
        "Python至少取得35个数据点": (
            markers["DATA_POINTS"] is not None
            and markers["DATA_POINTS"] >= 35
        ),
        "Python计算MA7与MA20": (
            markers["MA7"] is not None and markers["MA20"] is not None
        ),
        "Python计算RSI14和MACD": all(
            markers[x] is not None
            for x in ("RSI14", "MACD", "MACD_SIGNAL")
        ),
        "Python计算区间收益与最大回撤": all(
            markers[x] is not None
            for x in ("PERIOD_RETURN_PCT", "MAX_DRAWDOWN_PCT")
        ),
        "生成最终分析报告": bool(answer2),
    }
    core_passed = all(checks.values())
    return {
        "passed": core_passed,
        "strict_passed": core_passed and len(source_urls) >= 2,
        "checks": checks,
        "source_provenance_verified": len(source_urls) >= 2,
        "web_search_calls": len(searches),
        "web_extractor_calls": len(extractors),
        "code_interpreter_calls": len(codes),
        "sources_from_tools": source_urls,
        "urls_in_answer_unverified": answer_urls,
        "code_markers": markers,
        "code_logs": logs,
        "executed_code": executed_code,
        "clarification": answer1,
        "answer": answer2,
    }


def run(client, request, label):
    response = None
    counts = {}
    stream = None
    with tqdm(
        total=None, desc=f"{label}·流式事件", unit="事件",
        mininterval=0.5, dynamic_ncols=True, leave=False, position=1,
    ) as event_bar:
        try:
            tqdm.write(f"[开始] {label}，正在连接 Responses API")
            stream = client.responses.create(**request)
            for event in stream:
                kind = event.type
                counts[kind] = counts.get(kind, 0) + 1
                event_bar.update(1)
                if kind == "response.web_search_call.completed":
                    tqdm.write("[工具] 托管网络搜索完成")
                elif kind == "response.code_interpreter_call.completed":
                    tqdm.write("[工具] 托管 Python 计算完成")
                elif kind == "response.web_extractor_call.completed":
                    tqdm.write("[工具] 托管网页抽取完成")
                elif kind in {
                    "response.completed", "response.failed", "response.incomplete"
                }:
                    if getattr(event, "response", None) is not None:
                        response = as_dict(event.response)
        finally:
            if stream is not None:
                stream.close()
    if response is None:
        raise RuntimeError(f"{label}没有收到完整响应，请检查网络或 API 状态")
    return response, counts


def summarize_usage(first, second):
    return {
        key: sum(
            int((data.get("usage") or {}).get(key) or 0)
            for data in (first, second)
        )
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def main():
    parser = argparse.ArgumentParser(
        description="实验 1-3：BTC 分析（先澄清、再搜索计算，带 tqdm 进度条）"
    )
    parser.add_argument("--dry-run", action="store_true", help="预览两轮请求，不调用 API")
    parser.add_argument("--model", help="覆盖 config.yaml 中的模型，例如 qwen3.7-plus")
    parser.add_argument("--max-tokens", type=int, help="覆盖 config.yaml 中的 max_tokens")
    parser.add_argument("--interactive", action="store_true", help="第一轮澄清后手动输入用户回答")
    parser.add_argument("--reply", help="指定第二轮用户回答；默认使用 CoinGecko + 六类指标")
    parser.add_argument("--with-extractor", action="store_true", help="额外启用托管 web_extractor")
    parser.add_argument("--output", type=Path, help="指定 JSON 结果保存路径")
    args = parser.parse_args()

    cfg = settings()
    cfg["model"] = args.model or cfg["model"]
    if args.max_tokens is not None:
        cfg["max_output_tokens"] = args.max_tokens
    if cfg["max_output_tokens"] <= 0:
        parser.error("max_tokens 必须大于 0")

    print(f"项目根目录：{cfg['root']}")
    print(f"配置文件：{cfg['root'] / 'config.yaml'}")
    print(f"环境文件：{cfg['root'] / '.env'}")
    print(f"模型：{cfg['model']}")
    print(f"API 地址：{cfg['base_url']}/responses")
    print(f"最大输出 Token：{cfg['max_output_tokens']}")

    request1 = build_request(cfg, INITIAL_TASK, with_extractor=args.with_extractor)
    if args.dry_run:
        request2 = build_request(
            cfg, args.reply or DEFAULT_REPLY,
            previous_response_id="<首轮返回的response.id>",
            with_extractor=args.with_extractor,
        )
        print("\n[第一轮请求预览]")
        print(json.dumps(request1, ensure_ascii=False, indent=2))
        print("\n[第二轮请求预览]")
        print(json.dumps(request2, ensure_ascii=False, indent=2))
        return 0

    if not cfg["api_key"]:
        parser.error(".env 中未找到 DASHSCOPE_API_KEY")

    try:
        from openai import OpenAI
    except ImportError as exc:
        parser.error("缺少 OpenAI SDK，请运行：python -m pip install -U openai")
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=900,
        max_retries=1,
    )
    first, second = {}, {}
    events1, events2 = {}, {}
    request2 = None
    user_reply = None
    error = None

    with tqdm(total=2, desc="实验1-3·整体轮次", unit="轮", dynamic_ncols=True) as overall:
        try:
            first, events1 = run(client, request1, "第一轮：意图澄清")
            overall.update(1)
            first_answer = extract_answer(first)
            tqdm.write("\n[第一轮模型回答]\n" + (first_answer or "（没有文本回答）"))

            first_ok = (
                first.get("status") == "completed"
                and bool(first_answer)
                and ("?" in first_answer or "？" in first_answer)
                and not all_tool_items(first)
            )
            if not first_ok:
                error = "第一轮未按要求先澄清，或提前调用了工具；为保持实验有效性，停止第二轮。"
                tqdm.write("[停止] " + error)
            elif not first.get("id"):
                error = "第一轮没有返回 response.id，无法续接上下文。"
                tqdm.write("[停止] " + error)
            else:
                if args.interactive:
                    tqdm.write("\n请输入你对模型澄清问题的回答；直接回车则使用教程默认回答。")
                    user_reply = input("你的回答：").strip() or args.reply or DEFAULT_REPLY
                else:
                    user_reply = args.reply or DEFAULT_REPLY
                    tqdm.write("\n[自动补充需求]\n" + user_reply)

                request2 = build_request(
                    cfg, user_reply, previous_response_id=first["id"],
                    with_extractor=args.with_extractor,
                )
                second, events2 = run(client, request2, "第二轮：搜索与Python分析")
                overall.update(1)
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            tqdm.write("\n[运行中断] " + error)

    validation = analyze(first, second, request2)
    output = args.output or (
        Path(__file__).resolve().parent / "results" /
        ("bitcoin_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    )
    record = {
        "experiment": "1-3-bitcoin-clarification-hosted-tools",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": cfg["model"],
        "base_url": cfg["base_url"],
        "initial_task": INITIAL_TASK,
        "user_reply": user_reply,
        "first_turn": {
            "request": request1, "response": first, "events": events1,
        },
        "second_turn": {
            "request": request2, "response": second, "events": events2,
        },
        "usage": summarize_usage(first, second),
        "validation": validation,
        "error": error,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[验收]")
    for label, ok in validation["checks"].items():
        print(f"  {label}：{'通过' if ok else '未通过'}")
    print(f"\n 核心流程验收：{'通过' if validation['passed'] else '未通过'}")
    print(f"来源严格验收：{'通过' if validation['strict_passed'] else '未通过'}")
    print(f"工具元数据来源：{len(validation['sources_from_tools'])} 条")
    print(f"最终回答中的链接（未经工具来源验证）：{len(validation['urls_in_answer_unverified'])} 条")
    print(f"Token 用量：{record['usage']['total_tokens']}")
    if validation["answer"]:
        print("\n[最终分析报告]\n" + validation["answer"])
    if error:
        print("\n[错误] " + error)
    print(f"\n结果已保存：{output}")
    if not validation["strict_passed"]:
        print("提示：请检查 JSON；来源元数据缺失不等于没有调用搜索工具。")

    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
