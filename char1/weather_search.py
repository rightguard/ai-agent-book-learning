
import os
import json
import yaml
import requests
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from openai import OpenAI

# 读取配置
root = Path(__file__).resolve().parent.parent
load_dotenv(root / ".env")
config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.environ["QWEN_API_BASE_URL"]
)

# 查询深圳未来三天
def weather_search():
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 22.5431,
        "longitude": 114.0579,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_direction_10m_dominant",
        "timezone": "Asia/Shanghai",
        "forecast_days": 3
    }, timeout=20)
    r.raise_for_status()
    return {"source": r.url, "daily": r.json()["daily"]}

# 注册工具
tools = [{
    "type": "function",
    "function": {
        "name": "weather_search",
        "description": "查询深圳未来三天的天气预报",
        "parameters": {"type": "object", "properties": {}}
    }
}]

today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
messages = [
    {"role": "system", "content": f"今天是{today}。你是天气Agent，必须先调用工具，再根据结果回答。weather_code采用WMO标准，风向单位为度，风速为km/h。禁止编造数据。"},
    {"role": "user", "content": "查询深圳未来三天的天气、气温、风向、风速和数据来源。"}
]

# Agent 循环
search_count = 0
for step in range(config["agent"]["max_iterations"]):
    print(f"\n第{step + 1}步")

    response = client.chat.completions.create(
        model=config["model"]["name"],
        messages=messages,
        tools=tools,
        temperature=config["model"]["temperature"],
        extra_body={"enable_thinking": False}
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        print("[最终回答]", msg.content if search_count else "模型未调用天气工具")
        break

    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [t.model_dump() for t in msg.tool_calls]
    })

    for call in msg.tool_calls:
        print("[行动]", call.function.name, call.function.arguments)

        try:
            if call.function.name != "weather_search":
                raise ValueError("未知工具")
            if search_count >= config["search"]["max_iterations"]:
                raise ValueError("搜索次数已达上限")
            search_count += 1
            result = weather_search()
        except Exception as e:
            result = {"error": str(e)}

        print("[观察]", json.dumps(result, ensure_ascii=False)[:1000])

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
else:
    print("[结束] 达到最大迭代次数")
