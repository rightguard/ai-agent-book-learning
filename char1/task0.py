import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量
load_dotenv()

# 读取上一级目录的 config.yaml
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 初始化 Qwen 客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.getenv("QWEN_API_BASE_URL"),
)

# 调用模型
response = client.chat.completions.create(
    model=config["model"]["name"],
    temperature=config["model"]["temperature"],
    top_p=config["model"]["top_p"],
    max_tokens=config["model"]["max_tokens"],
    messages=[
        {"role": "user", "content": "你好"}
    ],
)

print(response.choices[0].message.content)