#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 2-3：真实 Prompt Cache 与六种 Agent 上下文管理模式。

单文件运行、真实 OpenAI 兼容 API、无模拟成绩。将两个研究问题分开：
A. controlled cache: 长且稳定的材料 + 唯一组标识 + 预热 + 多轮重复测量；
B. dependent ReAct: 下一步文件名/检索词只有读取上一步工具结果才知道，
   验证滑动窗口是否真正截断以及对任务的影响。

服务端没有 cached_tokens 时记作 None；缓存命中与成功率不是预设结论。
TTFT 计时是客户端接收第一个非空文本/思考/工具调用分片的时间。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fnmatch
import hashlib
import itertools
import json
import os
import random
import re
import secrets
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MODES = (
    "correct", "dynamic_system", "shuffled_tools", "dynamic_profile",
    "sliding_window", "text_format",
)
MODEL_DIR = "/mnt/e/ai-agent-book/qwen3_0.6b"
SYSTEM_AGENT = (
    "你是受限文件调查 Agent。只能使用 find、read_file、grep 读取当前实验工作区。"
    "必须根据上一轮返回的实际文件名或检索码决定下一步；未知的信息不能猜测。"
    "依次完成入口文件定位、入口读取、线索检索、下一文件读取、第二次线索检索和终点读取。"
    "每轮尽量只调用一个必要工具，等返回结果再决定下一步。"
    "完成后用一句话同时报告 SESSION_ID 与 FINAL_CODE；如果信息不足则明确说明。"
    "工具报错时可调整参数，但不要无意义重复请求。"
)
TOOL_ORDER = list(MODES)


def here_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / ".env").is_file() or (p / "config.yaml").is_file():
            return p
    return here.parents[2]


def config_env() -> dict:
    root = here_root()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    try:
        import yaml
        file = root / "config.yaml"
        obj = yaml.safe_load(file.read_text(encoding="utf-8-sig")) if file.exists() else {}
        return obj if isinstance(obj, dict) else {}
    except (ImportError, OSError, ValueError):
        return {}


def resolve_settings(args) -> dict:
    cfg = config_env()
    m = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    if args.provider == "local":
        return {"provider": "local", "model": args.model or "qwen3_0.6b",
                "url": args.base_url or "http://127.0.0.1:8000/v1", "key": "EMPTY"}
    if args.provider == "moonshot":
        return {"provider": "moonshot", "model": args.model or "kimi-k2.6",
                "url": args.base_url or "https://api.moonshot.cn/v1",
                "key": os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")}
    return {"provider": "dashscope",
            "model": args.model or m.get("name") or os.getenv("MODEL_NAME") or "qwen3.7-flash-2026-07-15",
            "url": args.base_url or os.getenv("QWEN_API_BASE_URL") or
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "key": os.getenv("DASHSCOPE_API_KEY")}


def digest(obj) -> str:
    if isinstance(obj, bytes):
        data = obj
    elif isinstance(obj, str):
        data = obj.encode("utf-8")
    else:
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def persist(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def tool_specs(group_id: str) -> list[dict]:
    """每组独立固定前缀。描述中的固定组标识也防止部分平台先序列化工具导致串组。"""
    marker = f"[缓存实验隔离标识 {group_id}；仅用于隔离请求组，与工具用途无关] "
    s = lambda desc: {"type": "string", "description": desc}
    def fn(name, desc, props, required):
        return {"type": "function", "function": {"name": name,
                "description": marker + desc,
                "parameters": {"type": "object", "properties": props, "required": required}}}
    return [
        fn("find", "递归查找实验目录中的文件，返回相对路径。",
           {"pattern": s("文件名通配符，例 entry_*.md"),
            "directory": s("可选相对搜索目录，默认当前实验目录")}, ["pattern"]),
        fn("read_file", "读取实验目录内一个文件的内容。",
           {"file_path": s("相对文件路径"),
            "offset": {"type": "integer", "description": "起始行，0 起"},
            "size": {"type": "integer", "description": "最多读取行数"}}, ["file_path"]),
        fn("grep", "在实验目录里的文件中按正则表达式检索行。",
           {"pattern": s("应从上一条工具结果中获取的精确检索码"),
            "file_path": s("可选文件相对路径；省略则检索所有实验文本")}, ["pattern"]),
    ]


def rotate_tools(specs: list[dict], step: int, mode: str) -> list[dict]:
    obj = copy.deepcopy(specs)
    if mode == "shuffled_tools":
        orders = list(itertools.permutations(range(len(obj))))
        # 三工具 6 种排列逐轮变化；超过 6 轮时不可避免重新出现某种顺序。
        obj = [obj[i] for i in orders[step % len(orders)]]
    return obj


def path_in(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("禁止访问实验目录之外的文件")
    return path


def tool_exec(tool: str, arguments: dict, base: Path) -> dict:
    try:
        if tool == "find":
            directory = path_in(base, str(arguments.get("directory") or "."))
            if not directory.is_dir():
                raise NotADirectoryError(str(directory))
            pattern = str(arguments.get("pattern", "*"))
            result = sorted(p.relative_to(base).as_posix() for p in directory.rglob("*")
                            if p.is_file() and not p.is_symlink() and fnmatch.fnmatch(p.name, pattern))
            return {"success": True, "files": result[:60], "truncated": len(result) > 60}
        if tool == "read_file":
            file = path_in(base, str(arguments["file_path"]))
            lines = file.read_text(encoding="utf-8").splitlines()
            start = max(0, int(arguments.get("offset") or 0))
            length = max(1, min(300, int(arguments.get("size") or 300)))
            return {"success": True, "path": file.relative_to(base).as_posix(),
                    "content": "\n".join(lines[start:start+length]), "total_lines": len(lines)}
        if tool == "grep":
            regex = re.compile(str(arguments["pattern"]), re.IGNORECASE)
            paths = ([path_in(base, str(arguments["file_path"]))] if arguments.get("file_path")
                     else [p for p in base.rglob("*") if p.is_file() and not p.is_symlink()
                           and p.suffix in (".md", ".txt")])
            matches = []
            for path in paths:
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if regex.search(line):
                        matches.append({"file": path.relative_to(base).as_posix(),
                                        "line": index, "text": line[:450]})
            return {"success": True, "matches": matches[:60], "truncated": len(matches) > 60}
        return {"success": False, "error": f"未知工具：{tool}"}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def make_agent_fixture(folder: Path) -> dict:
    """一次实验六组使用同一组非可猜测的文件线索；只在目标文件出现最终校验码。"""
    folder.mkdir(parents=True, exist_ok=True)
    code = secrets.token_hex(5).upper()
    entry = f"entry_{code.lower()}.md"
    second = f"notes/stage_two_{code.lower()}.md"
    final = f"reports/closing_{code.lower()}.md"
    session = f"SESSION-{secrets.token_hex(6).upper()}"
    answer = f"FINAL-{secrets.token_hex(7).upper()}"
    clue_a = f"FIND_A_{secrets.token_hex(6).upper()}"
    clue_b = f"FIND_B_{secrets.token_hex(6).upper()}"
    docs = {
        entry: ("# 入口凭证\n" + f"SESSION_ID={session}\n"
                f"下一步必须使用 grep 搜索检索码 {clue_a}；路径未知，不能凭猜测选择。\n"),
        "docs/catalog.md": ("# 唯一索引\n" + f"检索码 {clue_a} 指向第二阶段文件：{second}\n"
                            "这一条索引需要通过 grep 工具读取。\n"),
        second: ("# 第二阶段凭证\n" + f"再次使用 grep 搜索检索码 {clue_b}，定位最终文件。\n"),
        "docs/final_index.md": ("# 最终索引\n" + f"检索码 {clue_b} 指向最后文件：{final}\n"),
        final: ("# 最终核验结论\n" + f"FINAL_CODE={answer}\n"
                "结论：固定上下文有利于跨请求前缀复用；不能用缓存比例单独评价任务表现。\n"),
    }
    for relative, body in docs.items():
        path = path_in(folder, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return {"entry": entry, "second": second, "final": final,
            "session": session, "final_code": answer, "clue_a": clue_a, "clue_b": clue_b,
            "expected_tool_sequence": ["find", "read_file", "grep", "read_file", "grep", "read_file"]}


def task_text() -> str:
    return (
        "请对工作区完成一次分步证据链调查。第1步用 find 查找 entry_*.md 的真实文件名；"
        "第2步用 read_file 读取这个入口文件并记下 SESSION_ID；"
        "第3步根据入口文件提供的检索码用 grep 找到第二阶段文件路径；"
        "第4步用 read_file 读取该文件；第5步根据它提供的新检索码再用 grep 找到最终文件；"
        "第6步用 read_file 读取最终文件并获得 FINAL_CODE。"
        "文件路径、检索码和 FINAL_CODE 均不可推测，必须等工具结果返回后再做下一步。"
        "请一次只发起你已掌握全部参数的必要工具调用；最后同时报告 SESSION_ID 和 FINAL_CODE。"
    )


def usage_info(item) -> dict:
    if item is None:
        return {"prompt_tokens": None, "completion_tokens": None, "cached_tokens": None,
                "cached_reported": False, "raw_usage": None}
    data = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
    data = data if isinstance(data, dict) else {}
    details = data.get("prompt_tokens_details") or data.get("input_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = data.get("prompt_cache_hit_tokens")
    p = data.get("prompt_tokens")
    if p is None:
        p = data.get("input_tokens")
    c = data.get("completion_tokens")
    if c is None:
        c = data.get("output_tokens")
    return {"prompt_tokens": p, "completion_tokens": c, "cached_tokens": cached,
            "cached_reported": cached is not None, "raw_usage": data}


def is_reasoning_model(s: dict) -> bool:
    m = s["model"].lower()
    return any(x in m for x in ("kimi-k2.5", "kimi-k2.6", "kimi-k2.7", "kimi-k3"))


def api_stream(client, s: dict, messages: list[dict], specs: list[dict],
               max_tokens: int, allow_tools: bool = True) -> dict:
    opts = {"model": s["model"], "messages": messages, "tools": specs,
            "tool_choice": "auto" if allow_tools else "none", "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 1.0 if is_reasoning_model(s) else 0.2, "max_tokens": max_tokens}
    if s["provider"] == "dashscope":
        opts["extra_body"] = {"enable_thinking": False}
    started = time.perf_counter()
    text, reasoning, tc_map = [], [], {}
    elapsed, ttft = None, None
    usage = None
    response_id = None
    finished = None
    # 不做降级到非流式，也不伪造缓存值，避免影响 TTFT 的定义。
    for chunk in client.chat.completions.create(**opts):
        response_id = response_id or getattr(chunk, "id", None)
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            delta = choice.delta
            piece = getattr(delta, "content", None) or ""
            thought = getattr(delta, "reasoning_content", None) or ""
            calls = getattr(delta, "tool_calls", None) or []
            if (piece or thought or calls) and ttft is None:
                ttft = time.perf_counter() - started
            if piece:
                text.append(piece)
            if thought:
                reasoning.append(thought)
            for call in calls:
                index = call.index if call.index is not None else 0
                obj = tc_map.setdefault(index, {"id": None, "type": "function",
                                                "function": {"name": "", "arguments": ""}})
                if getattr(call, "id", None):
                    obj["id"] = call.id
                fn = getattr(call, "function", None)
                if fn:
                    obj["function"]["name"] += getattr(fn, "name", None) or ""
                    obj["function"]["arguments"] += getattr(fn, "arguments", None) or ""
            if choice.finish_reason:
                finished = choice.finish_reason
    elapsed = time.perf_counter() - started
    tool_calls = [tc_map[i] for i in sorted(tc_map)]
    for idx, call in enumerate(tool_calls):
        if not call["id"]:
            call["id"] = f"fallback_{int(started * 1000000)}_{idx}"
    return {"content": "".join(text), "reasoning_content": "".join(reasoning),
            "tool_calls": tool_calls, "id": response_id, "ttft_s": ttft,
            "elapsed_s": elapsed, "finish_reason": finished,
            "usage": usage_info(usage), "message_count": len(messages),
            "request_hash": digest({"messages": messages, "tools": specs, "tool_choice": opts["tool_choice"]})}


def flatten(history: list[dict]) -> str:
    parts = []
    for msg in history:
        role = msg["role"].upper()
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            payload = json.dumps(msg["tool_calls"], ensure_ascii=False)
            parts.append(f"ASSISTANT TOOL_CALLS: {payload}")
        else:
            parts.append(f"{role}: {msg.get('content') or ''}")
    return "\n".join(parts)


def trim_agent_history(history: list[dict], window: int) -> tuple[list[dict], dict]:
    """从完整历史裁剪整组 assistant+全部 tool 结果；保留初始问题作锚点。"""
    if not history:
        return [], {"removed": 0, "removed_tool_messages": 0, "truncated": False}
    assert history[0]["role"] == "user"
    groups = []
    i = 1
    while i < len(history):
        chunk = [history[i]]
        i += 1
        if chunk[0]["role"] == "assistant":
            while i < len(history) and history[i]["role"] == "tool":
                chunk.append(history[i])
                i += 1
        groups.append(chunk)
    selected = groups[-window:]
    kept = [copy.deepcopy(history[0])] + [copy.deepcopy(x) for part in selected for x in part]
    removed = len(history) - len(kept)
    removed_tools = sum(msg["role"] == "tool" for part in groups[:-window] for msg in part)
    return kept, {"removed": removed, "removed_tool_messages": removed_tools,
                  "truncated": removed > 0}


def trim_cache_history(history: list[dict], window: int) -> tuple[list[dict], dict]:
    """历史为 user/assistant 对交替，裁剪除首条用户消息外的旧对话对。"""
    if not history:
        return [], {"removed": 0, "truncated": False}
    initial = copy.deepcopy(history[0])
    pairs = [history[i:i+2] for i in range(1, len(history), 2)]
    tail = pairs[-window:]
    kept = [initial] + [copy.deepcopy(m) for pair in tail for m in pair]
    return kept, {"removed": len(history) - len(kept), "truncated": len(kept) < len(history)}


def context_messages(mode: str, system: str, history: list[dict], step: int,
                     slide_groups: int, stage: str) -> tuple[list[dict], dict]:
    if mode == "dynamic_system":
        # 每轮在长系统前缀之前改动，真正破坏同组跨轮的共同前缀。
        system = f"NOW={step:08d}; TIME={dt.datetime.now().isoformat()}\n" + system
    state = {"removed": 0, "removed_tool_messages": 0, "truncated": False}
    if mode == "sliding_window":
        if stage == "agent":
            history, state = trim_agent_history(history, slide_groups)
        else:
            history, state = trim_cache_history(history, slide_groups)
    if mode == "text_format":
        return [{"role": "system", "content": system},
                {"role": "user", "content": flatten(history)}], state
    prefix = [{"role": "system", "content": system}]
    if mode == "dynamic_profile":
        prefix.append({"role": "user", "content":
                       f"[动态账户资料] 剩余额度={10000-step}；本轮标识={step:08d}"})
    return prefix + copy.deepcopy(history), state


def stable_context(count: int, group_id: str) -> str:
    """固定且可核对的合成测试材料，避免随机无意义字符的缓存压测。"""
    tips = [
        ("前缀", "改变前置系统内容会缩短请求之间相同的 token 前缀"),
        ("工具", "工具定义及其顺序可能参与服务端的提示词序列化"),
        ("性能", "TTFT 受缓存、队列、网络、服务端负载等共同影响"),
        ("历史", "滑动窗口压缩历史的同时可能删除关键工具事实"),
        ("角色", "标准结构化 role/tool 消息适合表达工具调用的来源"),
        ("缓存", "缓存 Token 占比必须和 prompt Token 绝对数量一起解读"),
    ]
    lines = [f"[缓存对照组唯一标识: {group_id}]",
             "你是一名实验记录员。以下是合成的稳定技术笔记，编号固定且只用于学习缓存行为。",
             "只需简短回答最后一条用户问题，不要调用工具，不能伪造额外资料。"]
    for i in range(count):
        topic, fact = tips[i % len(tips)]
        lines.append(f"固定备忘录 {i+1:03d}｜主题={topic}｜内容={fact}。"
                     f"记录号={i+1:03d}，审核标记=R{(i * 37 + 17) % 997:03d}。")
    return "\n".join(lines)


def metrics_from_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"observed_requests": 0, "prompt_tokens": None, "cached_tokens": None,
                "cache_ratio": None, "mean_ttft_s": None, "ttfts_s": []}
    usages = [r["usage"] for r in rows if not r.get("error")]
    def add_if_complete(key):
        values = [u.get(key) for u in usages]
        return sum(values) if len(values) == len(rows) and all(v is not None for v in values) else None
    prompt = add_if_complete("prompt_tokens")
    cached = add_if_complete("cached_tokens")
    compl = add_if_complete("completion_tokens")
    ttfts = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    valid = (len(usages) == len(rows) and len(ttfts) == len(rows) and
             all(r.get("id") for r in rows) and not any(r.get("error") for r in rows))
    return {"observed_requests": len(rows), "valid_receipts": valid,
            "prompt_tokens": prompt, "completion_tokens": compl,
            "cached_tokens": cached, "cache_reported": cached is not None,
            "cache_ratio": cached / prompt if cached is not None and prompt else None,
            "cache_hits": sum(int((r["usage"].get("cached_tokens") or 0) > 0) for r in rows)
                          if cached is not None else None,
            "cache_misses": sum(int(r["usage"].get("cached_tokens") == 0) for r in rows)
                            if cached is not None else None,
            "ttfts_s": ttfts,
            "mean_ttft_s": statistics.mean(ttfts) if ttfts else None,
            "mean_elapsed_s": statistics.mean([r.get("elapsed_s", 0) for r in rows])}


def cache_trial(client, s: dict, args, mode: str, group: str, bar) -> dict:
    system = stable_context(args.cache_prefix_records, group)
    specs = tool_specs(group)
    initial = {"role": "user", "content": "实验任务：对每次提出的记录号简短确认一项事实；回答一句话即可。"}
    history = [initial]
    # correct 直接在同一个完整列表上追加；其他模式每轮重建。
    live_correct = [{"role": "system", "content": system}, copy.deepcopy(initial)]
    history.append({"role": "user", "content": "请开始；简短确认缓存实验规则。"})
    live_correct.append(copy.deepcopy(history[-1]))
    logs = []
    evicted_total = 0
    for step in range(args.cache_warmups + args.cache_rounds):
        if mode == "correct":
            ctx, changes = live_correct, {"removed": 0, "truncated": False}
        else:
            ctx, changes = context_messages(mode, system, history, step,
                                           args.sliding_window_groups, "cache")
        defs = rotate_tools(specs, step, mode)
        # 记录整个序列，确保没有跨组误复用或无法区分的动态改动。
        entry = {"step": step + 1, "warmup": step < args.cache_warmups,
                 "window": changes, "request_messages": copy.deepcopy(ctx),
                 "request_tools": copy.deepcopy(defs)}
        try:
            # 使用 auto 才能确定工具定义真实参与模型请求；none 可能让
            # 某些服务端直接剔除工具列表，使 shuffled_tools 消融无效。
            res = api_stream(client, s, ctx, defs, args.cache_max_tokens, allow_tools=True)
            entry.update(res)
            if res["tool_calls"]:
                entry["error"] = "缓存对照要求模型仅回答文本，但本次返回工具调用；该组无效"
            if not res.get("id") or res["ttft_s"] is None:
                entry["error"] = "缺少真实服务端响应 ID 或首分片计时"
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        logs.append(entry)
        bar.update(1)
        evicted_total += int(changes.get("removed", 0))
        if entry.get("error"):
            break
        assistant = {"role": "assistant", "content": res["content"]}
        history.append(assistant)
        query = {"role": "user", "content": f"第 {step+2} 次核验：请用一句话说明第 {step % args.cache_prefix_records+1:03d} 条固定备忘录的主题。"}
        history.append(query)
        if mode == "correct":
            live_correct.append(copy.deepcopy(assistant))
            live_correct.append(copy.deepcopy(query))
    measured = [x for x in logs if not x["warmup"]]
    m = metrics_from_rows(measured)
    m.update({"warmup_requests": min(args.cache_warmups, len(logs)),
              "expected_measured_requests": args.cache_rounds,
              "window_removed_messages": evicted_total,
              "first_request_cached_tokens": (logs[0].get("usage", {}).get("cached_tokens")
                                              if logs and "usage" in logs[0] else None),
              "measured_complete": len(measured) == args.cache_rounds and m.get("valid_receipts") is True})
    return {"summary": m, "trace": logs}


def make_api_client(s: dict, args):
    from openai import OpenAI
    return OpenAI(api_key=s["key"], base_url=s["url"],
                  timeout=args.timeout, max_retries=1)


def cache_suite(s: dict, args, modes: list[str], run_dir: Path) -> dict:
    from tqdm import tqdm
    rng = random.Random(args.seed)
    client = make_api_client(s, args)
    collected = {mode: [] for mode in modes}
    for replicate in range(args.cache_repeats):
        sequence = list(modes)
        rng.shuffle(sequence)
        print(f"\n[缓存对照] 第 {replicate+1}/{args.cache_repeats} 次重复；模式顺序：{', '.join(sequence)}")
        for mode in sequence:
            group_id = "G" + secrets.token_hex(12).upper()  # 模式和重复之间均不同
            print(f"  --- {mode} ---")
            with tqdm(total=args.cache_warmups + args.cache_rounds,
                      desc=f"cache-{mode}-r{replicate+1}", unit="请求") as bar:
                obs = cache_trial(client, s, args, mode, group_id, bar)
            obs["group_id"] = group_id
            obs["replicate"] = replicate + 1
            collected[mode].append(obs)
            persist(run_dir / "cache_trials" / f"{mode}_rep{replicate+1}.json", obs)
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
    aggregate = {}
    for mode in modes:
        trials = collected[mode]
        samples = [r for t in trials for r in t["trace"] if not r.get("warmup")]
        m = metrics_from_rows(samples)
        ratios = [t["summary"]["cache_ratio"] for t in trials]
        m.update({"trials_completed": sum(t["summary"]["measured_complete"] for t in trials),
                  "expected_trials": args.cache_repeats,
                  "per_trial_cache_ratio": ratios,
                  "cache_ratio_sd": statistics.stdev(ratios) if len(ratios) >= 2 and
                                    all(x is not None for x in ratios) else None,
                  "measured_complete": (len(trials) == args.cache_repeats and
                                        all(t["summary"]["measured_complete"] for t in trials)),
                  "first_request_cache_tokens": [t["summary"]["first_request_cached_tokens"]
                                                 for t in trials],
                  "window_removed_messages": sum(t["summary"]["window_removed_messages"] for t in trials)})
        aggregate[mode] = {"summary": m, "trials": trials}
    return aggregate


def agent_trial(client, s: dict, args, mode: str, base: Path, ground_truth: dict,
                group: str, bar) -> dict:
    specs = tool_specs(group)
    system = f"[独立 ReAct 上下文组标识: {group}]\n" + SYSTEM_AGENT
    user = {"role": "user", "content": task_text()}
    history = [copy.deepcopy(user)]
    correct_live = [{"role": "system", "content": system}, copy.deepcopy(user)]
    logs = []
    evicted_messages, evicted_tool_messages, truncations = 0, 0, 0
    final_reply = ""
    for step in range(args.max_iterations):
        if mode == "correct":
            ctx, changes = correct_live, {"removed": 0, "removed_tool_messages": 0,
                                          "truncated": False}
        else:
            ctx, changes = context_messages(mode, system, history, step,
                                            args.sliding_window_groups, "agent")
        defs = rotate_tools(specs, step, mode)
        entry = {"step": step + 1, "request_messages": copy.deepcopy(ctx),
                 "request_tools": copy.deepcopy(defs), "window": changes}
        if changes.get("truncated"):
            truncations += 1
        evicted_messages += changes.get("removed", 0)
        evicted_tool_messages += changes.get("removed_tool_messages", 0)
        try:
            out = api_stream(client, s, ctx, defs, args.max_tokens, allow_tools=True)
            entry.update(out)
            if not out.get("id") or out.get("ttft_s") is None:
                entry["error"] = "无有效服务端响应 ID 或首分片计时"
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        logs.append(entry)
        bar.update(1)
        if entry.get("error"):
            break
        tcalls = out["tool_calls"]
        assistant = {"role": "assistant", "content": out["content"] or "",
                     **({"tool_calls": tcalls} if tcalls else {})}
        if out.get("reasoning_content") and s["provider"] == "moonshot":
            assistant["reasoning_content"] = out["reasoning_content"]
        history.append(copy.deepcopy(assistant))
        if mode == "correct":
            correct_live.append(copy.deepcopy(assistant))
        if not tcalls:
            final_reply = out["content"] or ""
            break
        for number, call in enumerate(tcalls):
            fname = call["function"]["name"]
            try:
                params = json.loads(call["function"].get("arguments") or "{}")
                if not isinstance(params, dict):
                    raise ValueError("工具参数不是 JSON 对象")
            except Exception as exc:
                params = {"_parsing_error": str(exc)}
            result = (tool_exec(fname, params, base) if "_parsing_error" not in params
                      else {"success": False, "error": params["_parsing_error"]})
            entry.setdefault("tools_executed", []).append({"name": fname, "args": params,
                                                            "result": result})
            tool_message = {"role": "tool", "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False)}
            history.append(copy.deepcopy(tool_message))
            if mode == "correct":
                correct_live.append(copy.deepcopy(tool_message))
    tools_used = [e for r in logs for e in r.get("tools_executed", [])]
    successful = [e for e in tools_used if e["result"].get("success")]
    # 客观工具证据：需要实际读取每一个文件、检索每一个隐藏线索，而非只听模型声称。
    def successful_event(action: str, expected: str, field: str) -> bool:
        if field == "path":
            return any(e["name"] == action and e["result"].get("path") == expected
                       for e in successful)
        return any(e["name"] == action and expected in str(e["args"].get("pattern", ""))
                   and any(expected in x.get("text", "")
                           for x in e["result"].get("matches", [])) for e in successful)
    found_entry = any(e["name"] == "find" and ground_truth["entry"] in
                      e["result"].get("files", []) for e in successful)
    def _is_correct_stage(e: dict, stage: int) -> bool:
        if not e["result"].get("success"):
            return False
        if stage == 0:
            return e["name"] == "find" and ground_truth["entry"] in e["result"].get("files", [])
        if stage in (1, 3, 5):
            target = {1: "entry", 3: "second", 5: "final"}[stage]
            return e["name"] == "read_file" and e["result"].get("path") == ground_truth[target]
        target_code = ground_truth["clue_a" if stage == 2 else "clue_b"]
        return (e["name"] == "grep" and target_code in str(e["args"].get("pattern", ""))
                and any(target_code in hit.get("text", "") for hit in e["result"].get("matches", [])))
    stage_cursor = 0
    for event in tools_used:
        if stage_cursor < 6 and _is_correct_stage(event, stage_cursor):
            stage_cursor += 1
    evidence = {"entry_found": found_entry,
                "entry_read": successful_event("read_file", ground_truth["entry"], "path"),
                "first_clue_grepped": successful_event("grep", ground_truth["clue_a"], "grep"),
                "second_file_read": successful_event("read_file", ground_truth["second"], "path"),
                "second_clue_grepped": successful_event("grep", ground_truth["clue_b"], "grep"),
                "final_file_read": successful_event("read_file", ground_truth["final"], "path"),
                "tool_chain_in_required_order": stage_cursor == 6,
                "answer_contains_session": ground_truth["session"] in final_reply,
                "answer_contains_final": ground_truth["final_code"] in final_reply}
    seen = set()
    repeated = 0
    for e in tools_used:
        key = (e["name"], json.dumps(e["args"], ensure_ascii=False, sort_keys=True))
        repeated += int(key in seen)
        seen.add(key)
    m = metrics_from_rows(logs)
    task_succeeded = all(evidence.values()) and not any(r.get("error") for r in logs)
    m.update({"task_success": task_succeeded, "task_evidence": evidence,
              "task_outcome": "success" if task_succeeded else "unsatisfied_evidence_or_answer",
              "tools_called": len(tools_used), "tools_succeeded": len(successful),
              "repeated_tool_calls": repeated, "window_truncations": truncations,
              "window_removed_messages_total": evicted_messages,
              "window_evicted_tool_messages_total": evicted_tool_messages,
              "final_reply": final_reply,
              "agent_trace_complete": m.get("valid_receipts") is True,
              "sufficient_agent_turns": len(logs) >= args.min_agent_rounds,
              "sliding_window_exercised": truncations > 0 and evicted_tool_messages > 0
                                           if mode == "sliding_window" else None})
    return {"summary": m, "trace": logs}


def agent_suite(s: dict, args, modes: list[str], run_dir: Path) -> dict:
    from tqdm import tqdm
    client = make_api_client(s, args)
    folder = run_dir / "fixture"
    gt = make_agent_fixture(folder)
    # 保存客观评分的哈希，不泄漏给 agent；日志下的评分 oracle 是实验者可见的。
    persist(run_dir / "ground_truth.json", gt)
    order = list(modes)
    random.Random(args.seed ^ 421).shuffle(order)
    print("\n[多轮 ReAct] 模式顺序：", ", ".join(order))
    collected = {}
    for mode in order:
        group = "A" + secrets.token_hex(12).upper()
        print("  --- " + mode + " ---")
        with tqdm(total=args.max_iterations, desc="agent-" + mode, unit="请求") as bar:
            data = agent_trial(client, s, args, mode, folder, gt, group, bar)
        data["group_id"] = group
        collected[mode] = data
        persist(run_dir / "agent_trials" / f"result_{mode}.json", data)
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)
    return {key: collected[key] for key in modes}


def merge_results(modes: list[str], cache_data: dict | None,
                  agent_data: dict | None) -> dict:
    merged = {}
    for mode in modes:
        ca = cache_data.get(mode) if cache_data is not None else None
        ag = agent_data.get(mode) if agent_data is not None else None
        sc = ca["summary"] if ca else {}
        sa = ag["summary"] if ag else {}
        # 兼容先前的 summary 字段，又明确区分数据来源。
        merged[mode] = {
            "summary": {"mode": mode,
                        "status": ("complete" if
                                   (ca is None or sc.get("measured_complete")) and
                                   (ag is None or sa.get("agent_trace_complete")) else "partial"),
                        "cache_ratio": sc.get("cache_ratio"),
                        "prompt_tokens": sc.get("prompt_tokens"),
                        "cached_tokens": sc.get("cached_tokens"),
                        "cache_observed": sc.get("cache_reported"),
                        "ttft_mean_s": sc.get("mean_ttft_s"),
                        "task_success": sa.get("task_success"),
                        "agent_iterations": sa.get("observed_requests"),
                        "sliding_window_exercised": sa.get("sliding_window_exercised"),
                        "repeated_tool_calls": sa.get("repeated_tool_calls")},
            "cache_benchmark": ca,
            "react_agent": ag,
        }
    return merged


def make_graphs(data: dict, out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    generated = []
    names = list(data)
    cache_values = [data[n]["summary"].get("cache_ratio") for n in names]
    if any(x is not None for x in cache_values):
        fig, ax = plt.subplots(figsize=(11, 4.6))
        bars = ax.bar(names, [100*v if v is not None else 0 for v in cache_values])
        for i, v in enumerate(cache_values):
            if v is None:
                ax.text(i, 0.9, "N/A", ha="center", va="bottom")
            elif v == 0:
                ax.text(i, 0.9, "0%", ha="center", va="bottom")
            else:
                ax.text(i, v*100+0.6, f"{v*100:.1f}%", ha="center", va="bottom")
        ax.set_ylabel("Provider-reported cached / prompt tokens (%)")
        ax.set_title("Experiment 2-3: cache benchmark (after warmup, real 0 vs N/A)")
        ax.set_ylim(0, max(8, max((v or 0) for v in cache_values)*100*1.25+3))
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        image = out / "cache_ratio.png"
        fig.savefig(image, dpi=170)
        plt.close(fig)
        generated.append(image.name)
        times = [data[n]["summary"].get("ttft_mean_s") for n in names]
        if any(t is not None for t in times):
            fig, ax = plt.subplots(figsize=(11, 4.6))
            ax.bar(names, [t if t is not None else 0 for t in times])
            for i, value in enumerate(times):
                if value is None:
                    ax.text(i, 0.02, "N/A", ha="center")
            ax.set_ylabel("Measured mean TTFT (s)")
            ax.set_title("Experiment 2-3: TTFT by mode (post-warmup)")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            image = out / "cache_ttft.png"
            fig.savefig(image, dpi=170)
            plt.close(fig)
            generated.append(image.name)
    rounds = [data[n]["summary"].get("agent_iterations") for n in names]
    if any(x is not None for x in rounds):
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.bar(names, [r or 0 for r in rounds])
        ax.set_ylabel("Actual agent model requests")
        ax.set_title("Experiment 2-3: dependent ReAct, actual model rounds")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        image = out / "agent_rounds.png"
        fig.savefig(image, dpi=170)
        plt.close(fig)
        generated.append(image.name)
    return generated


def display_report(items: list[str], price: float) -> None:
    rows = []
    for pathstr in items:
        path = Path(pathstr)
        candidates = sorted(path.glob("**/comparison.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            data = json.loads(candidate.read_text(encoding="utf-8"))
            for mode, record in (data.get("results") or {}).items():
                summary = record.get("summary") or {}
                p = summary.get("prompt_tokens")
                c = summary.get("cached_tokens")
                rows.append({"source": str(candidate), "mode": mode,
                             "cached_tokens": c, "prompt_tokens": p,
                             "cache_ratio": summary.get("cache_ratio"),
                             "cached_billable_token_equiv": p - (1-price)*c
                             if p is not None and c is not None else None,
                             "TTFT_s": summary.get("ttft_mean_s"),
                             "agent_requests": summary.get("agent_iterations", summary.get("iterations")),
                             "agent_success": summary.get("task_success"),
                             "sliding_window_exercised": summary.get("sliding_window_exercised")})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def self_test() -> None:
    """纯离线契约与边界测试，不构造虚假的实验结果。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        oracle = make_agent_fixture(root)
        fs = tool_exec("find", {"pattern": "entry_*.md"}, root)
        assert oracle["entry"] in fs["files"]
        entry = tool_exec("read_file", {"file_path": oracle["entry"]}, root)
        assert oracle["session"] in entry["content"]
        first = tool_exec("grep", {"pattern": oracle["clue_a"]}, root)
        assert any(oracle["second"] in x["text"] for x in first["matches"])
        second = tool_exec("read_file", {"file_path": oracle["second"]}, root)
        assert oracle["clue_b"] in second["content"]
        other = tool_exec("grep", {"pattern": oracle["clue_b"]}, root)
        assert any(oracle["final"] in x["text"] for x in other["matches"])
        last = tool_exec("read_file", {"file_path": oracle["final"]}, root)
        assert oracle["final_code"] in last["content"]
        assert not tool_exec("read_file", {"file_path": "../outside"}, root)["success"]
        task = {"role": "user", "content": task_text()}
        hist = [task]
        for i in range(4):
            hist.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "id"+str(i), "type": "function", "function":
                 {"name": "find", "arguments": "{}"}}]})
            hist.append({"role": "tool", "tool_call_id": "id"+str(i), "content": "{}"})
        trimmed, diag = trim_agent_history(hist, 2)
        assert diag["truncated"] and diag["removed_tool_messages"] == 2
        assert len(trimmed) == 5 and trimmed[0]["role"] == "user"
        for j, x in enumerate(trimmed):
            if x["role"] == "tool":
                assert trimmed[j-1]["role"] == "assistant"
        assert usage_info({"prompt_tokens": 100,
                           "prompt_tokens_details": {"cached_tokens": 0}})["cached_reported"]
        assert usage_info({"prompt_tokens": 100})["cached_tokens"] is None
        for mode in MODES:
            ctx, _ = context_messages(mode, "FIXED", hist, 2, 2, "agent")
            assert ctx[0]["role"] == "system"
        assert len({digest(tool_specs("G"+str(i))) for i in range(6)}) == 6
        defs = tool_specs("G1")
        assert len({tuple(x["function"]["name"] for x in rotate_tools(defs, k, "shuffled_tools"))
                    for k in range(6)}) == 6
    print("[self-test] 已通过：多阶段真实文件工具、窗口裁剪、独立前缀、乱序、缓存零值/N-A 区分")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="实验 2-3：缓存性能与多轮 ReAct 上下文管理（真实 API，独立文件）")
    ap.add_argument("--provider", choices=["dashscope", "local", "moonshot"], default="dashscope")
    ap.add_argument("--base-url"); ap.add_argument("--model")
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--compare", action="store_true", help="完整六模式实验，默认也为六模式")
    ap.add_argument("--phase", choices=["all", "cache", "agent"], default="all")
    ap.add_argument("--cache-repeats", type=int, default=2, help="每模式独立重复数，默认2，建议论文统计用3以上")
    ap.add_argument("--cache-warmups", type=int, default=1, help="每重复的缓存预热请求数，默认1")
    ap.add_argument("--cache-rounds", type=int, default=4, help="每重复的正式缓存测量请求数，默认4")
    ap.add_argument("--cache-prefix-records", type=int, default=48,
                    help="固定前缀的合成知识条目数，默认48，可调节缓存门槛")
    ap.add_argument("--cache-max-tokens", type=int, default=None,
                    help="缓存压测每请求生成上限，Qwen默认160，Kimi默认4096")
    ap.add_argument("--max-iterations", type=int, default=14, help="ReAct 最大轮数，默认14")
    ap.add_argument("--min-agent-rounds", type=int, default=5, help="有效长链代理测试所需最少轮数")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="ReAct 每次回复上限，Qwen默认1200，Kimi默认4096")
    ap.add_argument("--sliding-window-groups", type=int, default=2,
                    help="滑窗保存的最近 assistant+tool 完整组数，默认2")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--pause-seconds", type=float, default=0, help="各组之间暂停时间")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--report", nargs="*", metavar="COMPARISON_JSON_OR_DIR")
    ap.add_argument("--cache-price-ratio", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.report is not None:
        display_report(args.report or [str(Path(__file__).parent / "results")], args.cache_price_ratio)
        return 0
    if args.cache_repeats < 1 or args.cache_warmups < 1 or args.cache_rounds < 2:
        ap.error("cache-repeats≥1、cache-warmups≥1、cache-rounds≥2")
    if args.cache_prefix_records < 12 or args.max_iterations < 2 or args.min_agent_rounds < 2:
        ap.error("cache-prefix-records≥12、max-iterations≥2、min-agent-rounds≥2")
    if args.sliding_window_groups < 1 or args.timeout <= 0 or args.pause_seconds < 0:
        ap.error("滑窗组数至少为1，超时须为正，暂停时间不能为负")
    if not (0 <= args.cache_price_ratio <= 1):
        ap.error("cache-price-ratio必须在0和1之间")
    s = resolve_settings(args)
    if args.max_tokens is None:
        args.max_tokens = 4096 if is_reasoning_model(s) else 1200
    if args.cache_max_tokens is None:
        args.cache_max_tokens = 4096 if is_reasoning_model(s) else 160
    if is_reasoning_model(s) and min(args.max_tokens, args.cache_max_tokens) < 4096:
        ap.error("Kimi 推理模型输出预算需至少4096，否则可能截断有效回复")
    selected = [args.mode] if args.mode else list(MODES)
    run_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results" / timestamp()
    config = {"provider": s["provider"], "model": s["model"], "api_base": s["url"],
              "modes": selected, "phase": args.phase,
              "cache_repeats": args.cache_repeats, "cache_warmups": args.cache_warmups,
              "cache_rounds": args.cache_rounds, "cache_prefix_records": args.cache_prefix_records,
              "max_iterations": args.max_iterations, "sliding_window_groups": args.sliding_window_groups,
              "seed": args.seed, "output_dir": str(run_dir), "model_directory_when_local": MODEL_DIR,
              "analysis_note": "服务端缓存上报为0不等于字段缺失；缓存收益与Agent任务效果分开统计。"}
    if args.dry_run:
        print(json.dumps({**config, "key_present": bool(s["key"]),
                          "estimated_cache_calls": len(selected)*args.cache_repeats*
                          (args.cache_warmups+args.cache_rounds) if args.phase != "agent" else 0,
                          "maximum_agent_calls": len(selected)*args.max_iterations
                          if args.phase != "cache" else 0}, ensure_ascii=False, indent=2))
        return 0
    if not s["key"]:
        ap.error("缺少API密钥：DashScope需DASHSCOPE_API_KEY；Moonshot需MOONSHOT_API_KEY")
    persist(run_dir / "config.json", config)
    print("[2-3]", json.dumps(config, ensure_ascii=False))
    cache_data = cache_suite(s, args, selected, run_dir) if args.phase in ("cache", "all") else None
    agent_data = agent_suite(s, args, selected, run_dir) if args.phase in ("agent", "all") else None
    results = merge_results(selected, cache_data, agent_data)
    cache_measured = (cache_data is None or all(r["summary"].get("measured_complete")
                                                   for r in cache_data.values()))
    cache_usage = (cache_data is None or all(r["summary"].get("cache_reported")
                                                for r in cache_data.values()))
    agent_receipts = (agent_data is None or all(r["summary"].get("agent_trace_complete")
                                                  for r in agent_data.values()))
    agent_rounds = (agent_data is None or all(r["summary"].get("sufficient_agent_turns")
                                                for r in agent_data.values()))
    sliding_exercised = (agent_data is None or "sliding_window" not in agent_data or
                         agent_data["sliding_window"]["summary"].get("sliding_window_exercised"))
    benchmark_signal = (cache_data is None or "correct" not in cache_data or
                        (cache_data["correct"]["summary"].get("cached_tokens") or 0) > 0)
    all_six_modes = set(selected) == set(MODES)
    gates = {"all_six_modes": all_six_modes, "cache_measurements_complete": cache_measured,
             "cache_usage_reported": cache_usage, "correct_cache_hit_observed": benchmark_signal,
             "agent_receipts_complete": agent_receipts,
             "agent_has_multiple_rounds": agent_rounds,
             "sliding_window_really_evicted_tool_results": bool(sliding_exercised)}
    # 自然任务失败仍是有效实验结果；真实缓存缺失/没有窗口截断却不能宣布假设验证成功。
    required = ["cache_measurements_complete", "cache_usage_reported", "correct_cache_hit_observed"] if cache_data else []
    if agent_data:
        required += ["agent_receipts_complete", "agent_has_multiple_rounds", "sliding_window_really_evicted_tool_results"]
    status = "complete" if all(gates[k] for k in required) and (all_six_modes or args.mode) else "partial"
    outcome = {"experiment": "2-3", "status": status, "config": config, "gates": gates,
               "results": results,
               "analysis": {"api_calls_are_real": True, "cache_stats_are_provider_reported": True,
                            "cache_vs_agent_separated": True,
                            "experimental_group_isolation": "unique id in system and tool descriptions per mode and replicate",
                            "agent_failure_is_valid_observation": True,
                            "cached_tokens_missing_is_not_zero": True,
                            "cold_request_cache_hits_are_recorded_but_not_in_primary_ratio": True}}
    persist(run_dir / "comparison.json", outcome)
    for mode, item in results.items():
        persist(run_dir / f"result_{mode}.json", item)
    figures = make_graphs(results, run_dir)
    hashes = {}
    for file in sorted(run_dir.rglob("*")):
        if file.is_file() and file.name != "manifest.json":
            hashes[file.relative_to(run_dir).as_posix()] = digest(file.read_bytes())
    manifest = {"experiment": "2-3", "status": status, "gates": gates,
                "observations": {"agent_success_by_mode":
                                 {m: (agent_data[m]["summary"]["task_success"] if agent_data else None)
                                  for m in selected},
                                 "cache_ratio_by_mode":
                                 {m: (cache_data[m]["summary"]["cache_ratio"] if cache_data else None)
                                  for m in selected},
                                 "images": figures}, "artifacts_sha256": hashes}
    persist(run_dir / "manifest.json", manifest)
    display_report([str(run_dir / "comparison.json")], args.cache_price_ratio)
    print("[2-3 验收]", status, json.dumps(gates, ensure_ascii=False))
    print("[输出]", run_dir)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
