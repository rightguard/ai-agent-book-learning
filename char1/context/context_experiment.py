import argparse, ast, copy, json, operator, os, re, yaml
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 模式、汇率等参数
MODES = ("full", "no_history", "no_reasoning", "no_tool_calls", "no_tool_results")
RATES = {"USD": Decimal("1"), "EUR": Decimal("0.92"),
         "GBP": Decimal("0.79"), "JPY": Decimal("149.50")}
CENT = Decimal("0.01")
EXPECTED = {"annual_usd": "9602895.73", "quarter_average_usd": "2400723.93"}

TASK = """某公司四个季度的营收如下：
- Q1：2,500,000 USD（250 万美元）
- Q2：2,100,000 EUR（210 万欧元）
- Q3：1,800,000 GBP（180 万英镑）
- Q4：380,000,000 JPY（3.8 亿日元）

请调用提供的货币换算工具，将所有非美元营收换算为美元；
再使用计算工具求出全年营收总额和季度平均营收（单位：美元）。
最终结果均保留两位小数，并简要列出每个季度的美元金额。
不得自行猜测汇率，必须以工具实际返回的结果为依据。"""

# 系统提示词
SYSTEM = ("你是财务分析 Agent。请使用工具完成换汇和计算，并用中文回答。"
          "工具中的汇率是示例，并非实时汇率，请使用工具中的示例。"
          "只能依据成功返回的工具结果报告换算数据；"
          "不得编造汇率或工具结果，证据不足时应明确说明。")

TOOLS = [
    {"type": "function", 
     "function": {
        "name": "convert_currency", 
        "description": "根据固定的示例汇率表进行货币换算。",
        "parameters": {"type": "object", 
                       "properties": {
                           "amount": {"type": "number", 
                                      "description": "待换算金额，填写实际数字，如 2100000"},
                           "from_currency": {"type": "string", 
                                             "enum": list(RATES), 
                                             "description": "原币种代码，如 EUR"},
                           "to_currency": {"type": "string", 
                                           "enum": list(RATES), 
                                           "description": "目标币种代码，汇总时使用 USD"}},
                        "required": ["amount", "from_currency", "to_currency"]
                        }
                }
    },

    {"type": "function", 
     "function": {
        "name": "calculate", 
        "description": "安全计算四则运算表达式；只支持数字、+、-、*、/ 和括号，不要使用千位分隔符或 sum()。",
        "parameters": {"type": "object", 
                       "properties": {
                           "expression": {
                               "type": "string", 
                               "description": "算术表达式，如 (2500000+2282608.70)/2"}
                               }, 
                        "required": ["expression"]
                        }
                }
    }
]

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

# 安全计算
def safe_calculate(expr):
    if not isinstance(expr, str) or len(expr) > 300:
        raise ValueError("表达式长度无效")
    tree = ast.parse(expr, mode="eval") # mode="eval" 表示只解析单个表达式，排除了赋值语句等完整程序语句

    if sum(1 for _ in ast.walk(tree)) > 70: # 统计语法树总共有多少个节点
        raise ValueError("表达式复杂度超过限制")

    def walk(node):
        if isinstance(node, ast.Expression): # ast.Expression是表达式语法树的根节点，如果当前节点是表达式的根节点，就继续处理它里面的内容
            return walk(node.body)
        
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): # ast.Constant表示常量
            return Decimal(str(node.value))
        
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)): # UnaryOp表示一元运算；node.op 表示 AST 语法树节点中的运算符
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        
        if isinstance(node, ast.BinOp) and type(node.op) in OPS: # 处理加减乘除，ast.BinOp表示二元运算，即左右各有一个操作数的运算
            value = OPS[type(node.op)](walk(node.left), walk(node.right)) # 找到运算函数，递归处理左右操作数
            if not value.is_finite():
                raise ValueError("超出计算范围")
            return value
        raise ValueError("表达式包含无效符号")

    value = walk(tree) # 从语法树根节点开始递归计算，得到最终结果
    return {
        "value": str(value),
        "rounded_2": str(value.quantize(CENT, rounding=ROUND_HALF_UP)) # ROUND_HALF_UP 表示通常所说的四舍五入，即恰好处于中间值时向远离零的方向舍入
    }

def call_tool(name, args):
    try:
        if name == "convert_currency":
            src = str(args["from_currency"]).upper()
            dst = str(args["to_currency"]).upper()
            amount = Decimal(str(args["amount"]))

            if not amount.is_finite() or abs(amount) > Decimal("1e15"):
                raise ValueError("超出计算范围")

            result = (amount*RATES[dst] / RATES[src]).quantize(CENT, rounding=ROUND_HALF_UP)

            return {
                "from_currency": src,
                "to_currency": dst,
                "converted_amount": float(result),
                "reference_rate": str(RATES[dst] / RATES[src]),
                "rate_type": "fixed_example_not_live"
            }
        if name == "calculate":
            return safe_calculate(args["expression"])
        raise ValueError("未知工具:" + name)
    except (ValueError, KeyError, TypeError, ArithmeticError, SyntaxError) as exc:
        return {"error": str(exc)}

def settings():
    file_path = Path(__file__).resolve()
    root = next((p for p in file_path.parents if (p/"config.yaml").is_file() and (p/".env").is_file()), None)

    if root is None:
        raise FileNotFoundError("无.yaml和.env文件，无法加载配置")
    load_dotenv(root/".env")

    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    key = os.getenv("DASHSCOPE_API_KEY")

    if not key:
        raise ValueError("未配置DASHSCOPE_API_KEY环境变量")

    base_url = (os.getenv("QWEN_API_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model_cfg = cfg.get("model") or {}
    agent_cfg = cfg.get("agent") or {}
    return root, key, base_url, {
    "model": model_cfg.get("name", "qwen3.7-flash-2026-07-15"),
    "temperature": float(model_cfg.get("temperature", 0.3)),
    "top_p": float(model_cfg.get("top_p", 0.9)),
    "max_tokens": int(model_cfg.get("max_tokens", 8192)),
    "max_iterations": int(agent_cfg.get("max_iterations", 10)),
    "max_tool_calls": int(agent_cfg.get("max_tool_calls", 20)),
}

def tool_call_data(msg):
    return [{"id": t.id, 
             "type": "function", 
             "function": {
                 "name": t.function.name, "arguments": t.function.arguments or "{}"}} for t in (msg.tool_calls or [])]


def run_mode(client, cfg, mode, limit):
    history = [{"role": "system", "content": SYSTEM}, 
               {"role": "user", "content": TASK}]
    turns, actions, answer, error = [], [], "", None
    status = "iteration_limit"
    

    for step in range(1, limit + 1):
        sent = copy.deepcopy(history[:2] if mode == "no_history" else history)
        request = {
    "model": cfg["model"],
    "messages": sent,
    "temperature": cfg["temperature"],
    "top_p": cfg["top_p"],
    "max_tokens": cfg["max_tokens"],
    "extra_body": {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 2048
    }
}
        
        if mode != "no_tool_calls":
            request.update(tools=TOOLS, tool_choice="auto", parallel_tool_calls=False)
        turn = {"step": step, "request": copy.deepcopy(request), "observations": []}

        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            status, error = "api_error", str(exc)
            turn["error"] = error
            turns.append(turn)
            break

        choice = response.choices[0]
        msg = choice.message
        reasoning = (getattr(msg, "reasoning_content", None)
                     or (getattr(msg, "model_extra", None) or {}).get("reasoning_content"))
        calls = tool_call_data(msg)

        turn["response"] = {"id": response.id, "finish_reason": choice.finish_reason,
                            "content": msg.content or "", "reasoning_content": reasoning,
                            "tool_calls": calls,
                            "usage": response.usage.model_dump() if response.usage else None}
        turns.append(turn)
        print(f"  第 {step} 轮：工具调用 {len(calls)} 次，返回推理内容={bool(reasoning)}")

        if choice.finish_reason == "length":
            status = "truncated"
            break

        if not calls:
            answer = (msg.content or "").strip()
            status = "completed" if answer else "empty_response"
            break

        if len(actions) + len(calls) > cfg["max_tool_calls"]:
            status = "tool_limit"
            print("已达到工具调用次数限制")
            break

        assistant_msg = {"role": "assistant", "content": msg.content or "", "tool_calls": calls}
        if reasoning and mode != "no_reasoning":
            assistant_msg["reasoning_content"] = reasoning
        history.append(assistant_msg)

        for call in calls:
            name, raw_args = call["function"]["name"], call["function"]["arguments"]
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是 JSON 对象")
                result = call_tool(name, args)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                args, result = {}, {"error": f"工具参数无效：{exc}"}
            
            observed = "" if mode == "no_tool_results" else json.dumps(result, ensure_ascii=False)
            turn["observations"].append({"name": name, 
                                         "arguments": args, "result": result,
                                         "visible_to_model": mode != "no_tool_results"})
            
            history.append({"role": "tool", "tool_call_id": call["id"], "content": observed})
            signature = name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
            actions.append(signature)
            print(f"    调用：{name}({args}) | 结果已隐藏={mode == 'no_tool_results'} | 返回={result}")

    required = {"EUR": 2100000, "GBP": 1800000, "JPY": 380000000}
    visible = {obs["result"].get("from_currency") for turn in turns
               for obs in turn["observations"] if obs["name"] == "convert_currency"
               and obs["visible_to_model"] and "error" not in obs["result"]
               and obs["result"].get("to_currency") == "USD"
               and float(obs["arguments"].get("amount", -1)) ==
               required.get(obs["result"].get("from_currency"))}
    if mode == "no_history":
        visible.clear()
    
    correct = status == "completed" and all(
        re.search(
            r"(?<!\d)" + re.escape(number) + r"(?!\d)",
            answer.replace(",", "")
    ) for number in EXPECTED.values()
)
    
    counts = Counter(actions)
    has_evidence = all(currency in visible for currency in required)

    calc_values = {
        obs["result"].get("rounded_2")
        for turn in turns
        for obs in turn["observations"]
        if obs["name"] == "calculate"
        and obs["visible_to_model"]
        and "error" not in obs["result"]
}

    calc_verified = (
        mode != "no_history"
        and all(v in calc_values for v in EXPECTED.values())
    )

    return {"mode": mode, 
            "status": status, 
            "error": error, 
            "iterations": len(turns),
            "tool_calls": len(actions),
            "repeated_tool_calls": sum(n - 1 for n in counts.values() if n > 1),
            "reasoning_turns": sum(bool(t.get("response", {}).get("reasoning_content")) for t in turns),
            "reasoning_in_requests": sum(any(m.get("reasoning_content") for m in t["request"]["messages"]) for t in turns),
            "answer_correct": correct, 
            "has_required_observations": has_evidence,
            "calculation_results_verified": calc_verified,
            "task_success": correct and has_evidence and calc_verified,
            "visible_conversions": sorted(v for v in visible if v),
            "answer": answer, 
            "turns": turns}


def main():
    parser = argparse.ArgumentParser(description="实验：Qwen上下文消融对照")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="选择实验组，默认全部五组")
    parser.add_argument("--max-iterations", type=int, help="每组最多迭代次数")
    parser.add_argument("--output", type=Path, help="指定 JSON 结果文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不调用模型")

    args = parser.parse_args()
    root, key, base_url, cfg = settings()
    limit = args.max_iterations or cfg["max_iterations"]

    if not 2 <= limit <= 30:
        parser.error("最大迭代次数必须在2至30之间")
    print(f"项目根目录：{root}\n模型：{cfg['model']}\nAPI 地址：{base_url}\n实验组：{args.modes}")

    if args.dry_run:
        return

    client = OpenAI(api_key=key, base_url=base_url, timeout=180, max_retries=1)
    results = []
    for mode in args.modes:
        print(f"\n[实验组] {mode}")
        results.append(run_mode(client, cfg, mode, limit))

    full = next((r for r in results if r["mode"] == "full"), None)
    if full:
        for row in results:
            if row["mode"] == "no_reasoning":
                row["reasoning_ablation_meaningful"] = bool(
                    full["reasoning_in_requests"] and row["reasoning_turns"]
                    and not row["reasoning_in_requests"])

    record = {"experiment": "1-1-context-ablation", 
              "model": cfg["model"],
              "provider": "dashscope", 
              "base_url": base_url,
              "created_at": datetime.now().astimezone().isoformat(),
              "root_config": str(root / "config.yaml"), 
              "root_env": str(root / ".env"),
              "task": TASK, 
              "rate_type": "fixed_example_not_live",
              "expected": EXPECTED, 
              "max_iterations": limit,
              "thinking": {"enable_thinking": True, "preserve_thinking": True,
                           "thinking_budget": 2048},
              "results": results}
    
    output = args.output or Path(__file__).resolve().parent / "results" / (
        "context_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n实验组            状态             调用   重复   数值正确  观测齐全  请求含历史推理")
    for row in results:
        print(f"{row['mode']:<16} {row['status']:<16} {row['tool_calls']:>5}  "
              f"{row['repeated_tool_calls']:>7}  {str(row['answer_correct']):<7}  "
              f"{str(row['has_required_observations']):<8}  {row['reasoning_in_requests']}")
    print(f"\n结果已保存：{output}")


if __name__ == "__main__":
    main()