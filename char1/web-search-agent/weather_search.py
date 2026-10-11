
import os, json, yaml, requests, sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI

# 读取配置
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["QWEN_API_BASE_URL"],
    timeout=120
)

CITIES = {"深圳": (22.5431, 114.0579), "广州": (23.1291, 113.2644)}
WMO = {
    0: "晴", 1: "大部晴", 2: "局部多云", 3: "阴", 45: "雾", 48: "结霜雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "轻冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "轻冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷雨", 96: "雷雨伴小冰雹", 99: "雷雨伴大冰雹",
}

# 查询广州、深圳未来三天天气
def weather_search(city):
    if city not in CITIES:
        raise ValueError(f"未知城市{city}")
    lat, lon = CITIES[city]

    response = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_direction_10m_dominant",
        "timezone": "Asia/Shanghai",
        "forecast_days": 3
    }, 
    timeout=(120, 120))
    response.raise_for_status()

    data = response.json()
    daily = data["daily"]
    if len(daily["time"]) != 3:
        raise ValueError("未返回三天的天气")
    daily["weather_text"] = [WMO.get(code, f"WMO {code}") for code in daily["weather_code"]]
    directions = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")
    daily["wind_direction_text"] = [
        directions[int((degree % 360 + 22.5) // 45) % 8] + "风"
        for degree in daily["wind_direction_10m_dominant"]
    ]
    return {
        "city": city,
        "queried_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source": response.url,
        "daily_units": data["daily_units"],
        "daily": daily,
    }

# 注册工具
TOOLS = [{
    "type": "function",
    "function": {
        "name": "weather_search",
        "description": "查询深圳或广州未来三天的天气、气温、风速、风向。",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "enum": list(CITIES)}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]

messages = [
    {"role": "system", "content": f"今天是{today}。你是天气Agent，必须先调用工具，再根据结果回答。weather_code采用WMO标准，风向单位为度，风速为km/h。禁止编造数据。数据来源必须使用原始URL，禁止嵌套Markdown链接。"},
    {"role": "user", "content": "查询广州和深圳未来三天的天气、气温、风向、风速和数据来源。"}
]

# 主运行函数
def main():
    question = " ".join(sys.argv[1:]) or "查询广州和深圳未来三天的天气、气温、风向、风速和数据来源。"
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    messages = [
        {"role": "system", "content": (
            f"今天是{today}。你是天气Agent，需要天气事实时先调用工具。"
            "根据用户的问题决定查询哪些城市。只根据工具结果回答，"
            "逐日说明天气、气温、风速和风向；风向角度表示风吹来的方向。"
            "注明数据来源和查询时间；遇到错误不得猜测。"
        )},
        {"role": "user", "content": question},
    ]
    searched = set()
    attempts = 0
    status, answer = "max_iterations", ""
    max_turns = int(cfg["agent"]["max_iterations"])
    max_searches = int(cfg["search"]["max_iterations"])

    # agent主循环
    for turn in range(1, max_turns + 1):
        try:
            response = client.chat.completions.create(
                model=cfg["model"]["name"], 
                messages=messages, 
                tools=TOOLS,
                tool_choice="auto", 
                parallel_tool_calls=False,
                temperature=cfg["model"].get("temperature", 0.2),
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:
            status, answer = "model_error", str(exc)
            break
        choice = response.choices[0]
        msg = choice.message
        print(f"\n[第{turn}轮] {choice.finish_reason}")

        # 没有工具调用就停止，agent的核心停止机制
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            status = "completed" if searched and answer and choice.finish_reason == "stop" else "no_valid_answer"
            messages.append({"role": "assistant", "content": answer})
            break

        # 追加对话
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, 
                "arguments": c.function.arguments,
            }} for c in msg.tool_calls],
        })

        for call in msg.tool_calls:
            name, raw = call.function.name, call.function.arguments or "{}"
            print(f"[行动] {name}({raw})")
            try:
                args = json.loads(raw)
                if name != "weather_search" or not isinstance(args, dict):
                    raise ValueError("未知工具或参数格式错误")
                city = args["city"]
                if city in searched:
                    raise ValueError("该城市已查询，请使用已有数据")
                if attempts >= max_searches:
                    raise ValueError("搜索次数达到上限")
                attempts += 1
                result = weather_search(city)
                searched.add(city)
            except Exception as exc:
                result = {"error": str(exc)}
            print("[观察]", json.dumps(result, ensure_ascii=False))
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    output = Path(__file__).with_name("weather_trace.json")
    output.write_text(json.dumps({
        "experiment": "1-2-weather-variant", 
        "model": cfg["model"]["name"],
        "question": question, 
        "status": status,
        "search_attempts": attempts, 
        "successful_cities": sorted(searched),
        "messages": messages, 
        "answer": answer,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[状态] {status}\n[最终回答] {answer or '未完成'}")
    print(f"[轨迹] {output}")


if __name__ == "__main__":
    main()

