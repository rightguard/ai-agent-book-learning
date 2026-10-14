
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

DEFAULT_MODEL_DIR = Path('/mnt/e/ai-agent-book/qwen3_0.6b')
DEFAULT_SERVED_NAME = 'qwen3_0.6b'
TASK = 'What are the current time and weather in Vancouver? Call both tools.'
SYSTEM = (
    'You are a careful tool-calling assistant. Do not invent live facts. '
    'For Vancouver time use America/Vancouver. For weather use Vancouver, Canada. '
    'The user wants time and current temperature; call get_current_time AND '
    'get_current_temperature in your first turn, in parallel when possible. '
    'After receiving the observations, give a concise answer. '
    'If a tool reports an error, report the unavailable value rather than inventing it.'
)
TOOL_SCHEMAS = [
    {'type': 'function', 'function': {
        'name': 'get_current_time',
        'description': 'Get current real date and local time for Vancouver, Canada.',
        'parameters': {'type': 'object', 'properties': {
            'timezone': {'type': 'string', 'description': 'IANA zone: America/Vancouver'}},
            'required': ['timezone']}}},
    {'type': 'function', 'function': {
        'name': 'get_current_temperature',
        'description': 'Get current Vancouver temperature from Open-Meteo. Never fabricate.',
        'parameters': {'type': 'object', 'properties': {
            'location': {'type': 'string', 'description': 'Vancouver, Canada'},
            'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}},
            'required': ['location']}}},
]
TOOL_TAGS = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def root_dir() -> Path:
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / 'config.yaml').is_file() or (parent / '.env').is_file():
            return parent
    return path.parent


def optional_config() -> dict:
    root = root_dir()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / '.env')
    except ImportError:
        pass
    config_path = root / 'config.yaml'
    if config_path.is_file():
        try:
            import yaml
            return (yaml.safe_load(config_path.read_text(encoding='utf-8-sig')) or {}).get('experiment_2_1') or {}
        except ImportError:
            print('[提示] 未安装 pyyaml，采用脚本默认值', file=sys.stderr)
    return {}


def assert_local_url(url: str) -> None:
    obj = urlparse(url)
    if obj.scheme != 'http' or obj.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('禁止连接云端模型。本实验服务地址只允许 http://localhost / 127.0.0.1 / [::1]')


def launch_command(path: Path) -> str:
    return (f'vllm serve {path} --served-model-name {DEFAULT_SERVED_NAME} '
            '--host 127.0.0.1 --port 8000 --enable-auto-tool-choice '
            '--tool-call-parser hermes --max-model-len 8192')


def get_current_time(timezone: str) -> dict:
    if timezone != 'America/Vancouver':
        return {'error': 'timezone 应为 America/Vancouver'}
    return {'timezone': timezone, 'datetime': datetime.now(ZoneInfo(timezone)).isoformat()}


def get_current_temperature(location: str, unit: str = 'celsius') -> dict:
    if 'vancouver' not in location.lower():
        return {'error': '实验任务要求 Vancouver, Canada'}
    if unit not in ('celsius', 'fahrenheit'):
        return {'error': '非法 unit'}
    import requests
    url = 'https://api.open-meteo.com/v1/forecast'
    try:
        resp = requests.get(url, params={
            'latitude': 49.2827, 'longitude': -123.1207,
            'current': 'temperature_2m', 'timezone': 'America/Vancouver',
            'temperature_unit': unit,
        }, timeout=(8, 25))
        resp.raise_for_status()
        data = resp.json()
        current = data.get('current') or {}
        if current.get('temperature_2m') is None:
            raise ValueError('服务端没有返回实时温度')
        return {'location': 'Vancouver, Canada', 'temperature': current['temperature_2m'],
                'unit': (data.get('current_units') or {}).get('temperature_2m', ''),
                'observed_at': current.get('time'), 'source': resp.url}
    except Exception as exc:
        return {'error': f'天气 API 失败：{type(exc).__name__}: {exc}', 'source': url}


def run_one_tool(call: dict) -> dict:
    start = time.perf_counter()
    try:
        args = call.get('arguments') or {}
        args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(args, dict):
            raise ValueError('工具参数不是 JSON 对象')
        if call['name'] == 'get_current_time':
            val = get_current_time(**args)
        elif call['name'] == 'get_current_temperature':
            val = get_current_temperature(**args)
        else:
            val = {'error': f'未登记的工具：{call["name"]}'}
    except Exception as exc:
        val = {'error': f'{type(exc).__name__}: {exc}'}
    return {'call': call, 'result': val, 'duration_s': time.perf_counter() - start}


def parallel_tools(calls: list[dict]) -> dict:
    start = time.perf_counter()
    if not calls:
        return {'execution': 'none', 'wall_s': 0.0, 'items': []}
    with ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
        # executor.map 保证结果顺序与工具调用请求相同，但任务确实同时执行。
        observations = list(pool.map(run_one_tool, calls))
    return {'execution': 'ThreadPoolExecutor', 'wall_s': time.perf_counter()-start, 'items': observations}


def _to_json_object(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def parse_xml_calls(text: str) -> list[dict]:
    calls = []
    for index, match in enumerate(TOOL_TAGS.finditer(text)):
        obj = _to_json_object(match.group(1))
        if not obj or not isinstance(obj.get('name'), str):
            raise ValueError('原始 <tool_call> 缺少有效的 name 和 JSON 对象')
        args = obj.get('arguments', {})
        if not isinstance(args, dict):
            raise ValueError('原始 <tool_call> 的 arguments 必须为 JSON 对象')
        calls.append({'id': f'raw_{index}', 'name': obj['name'],
                      'arguments': json.dumps(args, ensure_ascii=False)})
    return calls


class LocalClient:
    def __init__(self, backend: str, base_url: str, model: str, timeout: int = 240):
        self.backend, self.base_url, self.model = backend, base_url.rstrip('/'), model
        self.timeout = timeout

    def models(self) -> list[str]:
        import requests
        path = '/models' if self.backend == 'vllm' else '/api/tags'
        response = requests.get(self.base_url + path, timeout=(5, 10))
        response.raise_for_status()
        items = response.json().get('data' if self.backend == 'vllm' else 'models', [])
        return [obj['id' if self.backend == 'vllm' else 'name'] for obj in items]

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None,
                    max_tokens: int = 256, on_delta=None) -> dict:
        import requests
        start = time.perf_counter()
        first = None
        pieces, reasoning, chunks, calls = [], [], [], {}
        usage = None
        stop_reason = None
        if self.backend == 'vllm':
            payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                       'max_tokens': max_tokens, 'stream': True,
                       'stream_options': {'include_usage': True}}
            if tools:
                payload.update({'tools': tools, 'tool_choice': 'auto', 'parallel_tool_calls': True})
            endpoint = '/chat/completions'
        else:
            payload = {'model': self.model, 'messages': messages, 'stream': True,
                       'think': True, 'keep_alive': '10m',
                       'options': {'temperature': 0, 'num_predict': max_tokens, 'num_ctx': 8192}}
            if tools:
                payload['tools'] = tools
            endpoint = '/api/chat'
        with requests.post(self.base_url + endpoint, json=payload, stream=True,
                           timeout=(10, self.timeout)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if self.backend == 'vllm':
                    if not line.startswith(b'data:'):
                        continue
                    packet = line[5:].strip()
                    if packet == b'[DONE]':
                        break
                    obj = json.loads(packet)
                    chunks.append(obj)
                    if obj.get('usage'):
                        usage = obj['usage']
                    choices = obj.get('choices') or []
                    if not choices:
                        continue
                    if choices[0].get('finish_reason'):
                        stop_reason = choices[0]['finish_reason']
                    delta = choices[0].get('delta') or {}
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                    fragments = delta.get('tool_calls') or []
                    for c in fragments:
                        index = c.get('index', 0)
                        item = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                        if c.get('id'):
                            item['id'] = c['id']
                        func = c.get('function') or {}
                        item['name'] += func.get('name') or ''
                        item['arguments'] += func.get('arguments') or ''
                else:
                    obj = json.loads(line)
                    chunks.append(obj)
                    message = obj.get('message') or {}
                    content, thought = message.get('content') or '', message.get('thinking') or ''
                    fragments = message.get('tool_calls') or []
                    for c in fragments:
                        index = len(calls)
                        func = c.get('function') or {}
                        args = func.get('arguments', {})
                        calls[index] = {'id': f'ollama_{index}', 'name': func.get('name', ''),
                                        'arguments': json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict) else str(args)}
                    if obj.get('done'):
                        usage = {'prompt_tokens': obj.get('prompt_eval_count'),
                                 'completion_tokens': obj.get('eval_count'),
                                 'server_eval_duration_ns': obj.get('eval_duration'),
                                 'server_prompt_eval_duration_ns': obj.get('prompt_eval_duration')}
                        stop_reason = obj.get('done_reason') or 'done'
                if first is None and (content or thought or fragments):
                    first = time.perf_counter() - start
                if content:
                    pieces.append(content)
                    if on_delta:
                        on_delta('content', content)
                if thought:
                    reasoning.append(thought)
                    if on_delta:
                        on_delta('thinking', thought)
        wall = time.perf_counter()-start
        completion = (usage or {}).get('completion_tokens')
        if self.backend == 'ollama':
            decode_s = (usage or {}).get('server_eval_duration_ns')
            decode_s = decode_s / 1e9 if decode_s else None
            speed_method = 'server eval_count / eval_duration'
        else:
            decode_s = wall-first if first is not None and wall > first else None
            speed_method = 'SSE completion_tokens / (wall - TTFT); includes transmission overhead'
        rate = completion/decode_s if isinstance(completion, (int, float)) and decode_s else None
        tool_calls = [
            {'id': c['id'] or f'call_{i}', 'name': c['name'], 'arguments': c['arguments']}
            for i, c in sorted(calls.items())
        ]
        return {'response': ''.join(pieces), 'thinking': ''.join(reasoning), 'tool_calls': tool_calls,
                'ttft_s': first, 'wall_s': wall, 'usage': usage, 'tokens_per_second': rate,
                'throughput_method': speed_method, 'finish_reason': stop_reason,
                'raw_chunks': chunks}


def read_local_server_identity(client: LocalClient, model_dir: Path) -> dict:
    """从本地服务的模型列表核对身份；不把名字相同误当作已验证权重。"""
    import requests
    result = {'backend': client.backend, 'requested_model': client.model,
              'identity_verified': False, 'model_digest': None}
    try:
        if client.backend == 'ollama':
            tags = requests.get(client.base_url + '/api/tags', timeout=(5, 15))
            tags.raise_for_status()
            selected = next((x for x in tags.json().get('models', [])
                             if x.get('name') == client.model), None)
            result['tag'] = selected
            digest = (selected or {}).get('digest')
            result['model_digest'] = digest
            result['identity_verified'] = bool(isinstance(digest, str) and len(digest) >= 16)
            try:
                ver = requests.get(client.base_url + '/api/version', timeout=(5, 15))
                if ver.ok:
                    result['server_version'] = ver.json()
            except requests.RequestException as exc:
                result['version_error'] = str(exc)
            try:
                show = requests.post(client.base_url + '/api/show', json={'model': client.model},
                                     timeout=(5, 20))
                if show.ok:
                    obj = show.json()
                    result['show_details'] = obj.get('details')
                    result['show_model_info'] = obj.get('model_info')
            except requests.RequestException as exc:
                result['model_detail_error'] = str(exc)
        else:
            response = requests.get(client.base_url + '/models', timeout=(5, 15))
            response.raise_for_status()
            selected = next((x for x in response.json().get('data', [])
                             if x.get('id') == client.model), None)
            root = (selected or {}).get('root')
            result['served_model_entry'] = selected
            result['served_root'] = root
            # root is normally the actual local model folder when vllm serve /path was used.
            result['identity_verified'] = bool(root and Path(root).is_dir() and
                                               Path(root).resolve() == model_dir.resolve())
        result['identity_caveat'] = (
            '验证依据来自本地服务报告；Ollama digest 可标识运行模型，但不等于 HF 目录哈希。'
            if client.backend == 'ollama' else
            'vLLM served_root 未验证时，请查看 vllm serve 启动命令确认本地权重路径。'
        )
    except (requests.RequestException, ValueError) as exc:
        result['identity_error'] = f'{type(exc).__name__}: {exc}'
    return result


def streaming_printer():
    current = [None]
    def print_delta(kind: str, text: str):
        if kind != current[0]:
            print('\n[' + ('模型思考' if kind == 'thinking' else '模型回答') + '] ', end='', flush=True)
            current[0] = kind
        sys.stdout.write(text)
        sys.stdout.flush()
    return print_delta


def tool_roundtrip(client: LocalClient, turns: int, max_tokens: int, tqdm, show_stream=True) -> dict:
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TASK}]
    trace, seen, final_answer = [], [], ''
    first_turn_parallel = False
    state = 'max_turns'
    with tqdm(total=turns, desc='2-1 ReAct 循环', unit='轮', dynamic_ncols=True) as progress:
        for i in range(turns):
            response = client.stream_chat(messages, TOOL_SCHEMAS, max_tokens,
                                          on_delta=streaming_printer() if show_stream else None)
            if show_stream:
                print()
            record = {'turn': i+1, 'messages': list(messages), 'response': response}
            trace.append(record)
            progress.update(1)
            calls = response['tool_calls']
            # 禁止将缺失/无效的 JSON 参数悄悄替换为 {} 并算作模型调用成功。
            if any(not c.get('name') or _to_json_object(c.get('arguments', '')) is None
                   for c in calls):
                state = 'invalid_model_tool_arguments'
                record['invalid_tool_calls'] = calls
                break
            if not calls:
                final_answer = response['response'].strip()
                state = 'answered' if final_answer else 'empty_final_answer'
                break
            names = [c['name'] for c in calls]
            print(f'[第 {i+1} 轮工具请求] {names}')
            if i == 0 and sorted(names) == ['get_current_temperature', 'get_current_time']:
                first_turn_parallel = True
            if len(seen) + len(calls) > 8:
                state = 'tool_limit'
                break
            seen.extend(names)
            if client.backend == 'vllm':
                messages.append({'role': 'assistant', 'content': response['response'], 'tool_calls': [
                    {'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
                    for c in calls]})
            else:
                messages.append({'role': 'assistant', 'content': response['response'], 'tool_calls': [
                    {'function': {'name': c['name'], 'arguments': _to_json_object(c['arguments']) or {}}}
                    for c in calls]})
            observations = parallel_tools(calls)
            record['observations'] = observations
            for item in observations['items']:
                result = item['result']
                print(f'  [工具反馈] {item["call"]["name"]}: {json.dumps(result, ensure_ascii=False)[:240]}')
                if client.backend == 'vllm':
                    messages.append({'role': 'tool', 'tool_call_id': item['call']['id'],
                                     'content': json.dumps(result, ensure_ascii=False)})
                else:
                    messages.append({'role': 'tool', 'tool_name': item['call']['name'],
                                     'content': json.dumps(result, ensure_ascii=False)})
    observations = [i for item in trace for i in (item.get('observations') or {}).get('items', [])]
    valid = {i['call']['name'] for i in observations if isinstance(i['result'], dict) and 'error' not in i['result']}
    arguments_by_name = {obs['call']['name']: _to_json_object(obs['call']['arguments']) or {}
                         for obs in observations}
    checks = {
        'model_responses_received': bool(trace),
        'exact_two_tools_first_turn': first_turn_parallel,
        'required_time_timezone': arguments_by_name.get('get_current_time', {}).get('timezone') == 'America/Vancouver',
        'required_weather_location': 'vancouver' in str(arguments_by_name.get('get_current_temperature', {}).get('location', '')).lower(),
        'both_tool_results_real': {'get_current_time', 'get_current_temperature'} <= valid,
        'actually_parallel_executed': bool(trace and trace[0].get('observations', {}).get('execution') == 'ThreadPoolExecutor'),
        'final_answer_after_feedback': state == 'answered' and len(observations) >= 2,
        'no_parser_fallback': True,  # 从未实现或启用替模型做决定的解析兜底
    }
    return {'status': state, 'passed': all(checks.values()), 'checks': checks,
            'final_answer': final_answer, 'requested_tool_names': seen, 'trace': trace}


def load_local_tokenizer(model_dir: Path):
    """只从本地模型文件载入分词器，用于恢复模型真正看到的 Chat Template。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('原始 Token 测试需要本地 tokenizer：python -m pip install transformers') from exc
    if not model_dir.is_dir():
        raise FileNotFoundError(f'本地模型目录不存在：{model_dir}')
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def raw_generate_service(client: LocalClient, prompt: str, max_tokens: int, tqdm,
                         show_stream: bool = False) -> dict:
    """向正在运行的本地引擎请求真实原始序列；不在脚本中重复加载模型权重。

    Ollama: /api/generate raw=true
    vLLM:   /v1/completions (NOT /v1/chat/completions); Qwen Chat Template
            由本地 tokenizer 生成，skip_special_tokens=false 保留原始标签。
    """
    import requests
    start = time.perf_counter()
    first = None
    pieces, chunks = [], []
    usage = None
    finish_reason = None
    if client.backend == 'ollama':
        endpoint = '/api/generate'
        request = {
            'model': client.model, 'prompt': prompt, 'raw': True, 'stream': True,
            'think': False, 'keep_alive': '10m',
            'options': {'temperature': 0, 'num_predict': max_tokens, 'num_ctx': 8192, 'seed': 21},
        }
    else:
        endpoint = '/completions'
        request = {
            'model': client.model, 'prompt': prompt, 'stream': True,
            'stream_options': {'include_usage': True}, 'temperature': 0,
            'max_tokens': max_tokens, 'skip_special_tokens': False,
            'spaces_between_special_tokens': False,
        }
    with tqdm(total=None, desc=f'2-1 {client.backend} 原始流', unit='片段', dynamic_ncols=True) as bar:
        with requests.post(client.base_url + endpoint, json=request,
                           stream=True, timeout=(10, client.timeout)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if client.backend == 'ollama':
                    obj = json.loads(line)
                    chunks.append(obj)
                    piece = obj.get('response') or ''
                    if obj.get('done'):
                        finish_reason = obj.get('done_reason') or 'done'
                        usage = {'prompt_tokens': obj.get('prompt_eval_count'),
                                 'completion_tokens': obj.get('eval_count'),
                                 'eval_duration_ns': obj.get('eval_duration')}
                else:
                    if not line.startswith(b'data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == b'[DONE]':
                        break
                    obj = json.loads(payload)
                    chunks.append(obj)
                    if obj.get('usage'):
                        usage = obj['usage']
                    choices = obj.get('choices') or []
                    choice = choices[0] if choices else {}
                    piece = choice.get('text') or ''
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                if piece:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(piece)
                    bar.update(1)
                    if show_stream:
                        sys.stdout.write(piece)
                        sys.stdout.flush()
    if show_stream:
        print()
    text = ''.join(pieces)
    return {
        'backend': client.backend, 'endpoint': client.base_url + endpoint,
        'request': request, 'prompt': prompt, 'rendered_prompt_sha256': sha256_text(prompt),
        'raw_text': text, 'text': text, 'text_sha256': sha256_text(text),
        'raw_chunks': chunks, 'ttft_s': first, 'wall_s': time.perf_counter() - start,
        'usage': usage, 'finish_reason': finish_reason, 'streamed_nonempty': bool(text),
        'raw_mode': ('Ollama generate raw=true' if client.backend == 'ollama'
                     else 'vLLM completions skip_special_tokens=false'),
    }


def run_raw_experiment(client: LocalClient, model_dir: Path, engine: str, device: str,
                       max_tokens: int, tqdm, show_stream: bool = True) -> dict:
    """真正原始输入/输出 + 并行工具 + 观察回灌 + 第二轮最终回答。

    判定只统计模型原样生成的工具调用。解析失败/缺少调用均不替模型兜底。
    """
    if engine == 'skip':
        return {'status': 'skipped', 'passed': False,
                'reason': '明确跳过原始 Token 流，无法完成完整 2-1 实验'}
    if engine not in ('service', 'ollama', 'vllm'):
        raise ValueError(f'不支持的 raw engine: {engine}')
    if engine in ('ollama', 'vllm') and engine != client.backend:
        raise ValueError(f'--raw-engine {engine} 与 --backend {client.backend} 不匹配')
    tokenizer = load_local_tokenizer(model_dir)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TASK}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    print('\n[原始模型输出 / 第一轮]' ) if show_stream else None
    first = raw_generate_service(client, prompt, max_tokens, tqdm, show_stream=show_stream)
    calls = parse_xml_calls(first['raw_text'])
    if calls:
        parallel = parallel_tools(calls)
        messages.append({'role': 'assistant', 'content': first['raw_text']})
        for item in parallel['items']:
            messages.append({'role': 'tool',
                             'content': json.dumps(item['result'], ensure_ascii=False)})
        second_prompt = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        print('\n[原始模型输出 / 回灌工具后第二轮]') if show_stream else None
        second = raw_generate_service(client, second_prompt, max_tokens, tqdm, show_stream=show_stream)
        second_calls = parse_xml_calls(second['raw_text'])
    else:
        parallel = None
        second_prompt, second, second_calls = None, None, []
    names = [x['name'] for x in calls]
    by_name = {x['name']: _to_json_object(x['arguments']) or {} for x in calls}
    checks = {
        'chat_template_visible': all(t in prompt for t in ('<|im_start|>', '<|im_end|>', '<tools>')),
        'raw_think_marker_observed': '<think>' in first['raw_text'],
        'raw_tool_call_tags_observed': '<tool_call>' in first['raw_text'],
        'first_turn_exact_two_tools': (len(names) == 2 and
                                       sorted(names) == ['get_current_temperature', 'get_current_time']),
        'time_zone_matches': by_name.get('get_current_time', {}).get('timezone') == 'America/Vancouver',
        'weather_location_matches': 'vancouver' in str(by_name.get('get_current_temperature', {}).get('location', '')).lower(),
        'parallel_tools_ran': parallel is not None and parallel.get('execution') == 'ThreadPoolExecutor',
        'tool_results_valid': parallel is not None and len(parallel['items']) == 2 and
                              all('error' not in item['result'] for item in parallel['items']),
        'second_turn_final_answer': bool(second and second['raw_text'].strip()) and not second_calls,
        'real_stream_chunks_retained': bool(first['raw_chunks']) and bool(second and second['raw_chunks']),
    }
    return {
        'status': 'passed' if all(checks.values()) else 'incomplete',
        'passed': all(checks.values()), 'engine': client.backend, 'checks': checks,
        'first_rendered_prompt': prompt, 'first_turn': first,
        'parsed_first_turn_calls': calls, 'parallel_execution': parallel,
        'second_rendered_prompt': second_prompt, 'second_turn': second,
        'second_turn_tool_calls': second_calls,
        'canonical_raw_passed': all(checks.values()),
        'protocol_note': ('此路径完全使用本地服务的原始生成接口；官方严格规范的服务身份为 Ollama。'
                          if client.backend == 'vllm' else 'Ollama 原生 raw=true 路径。'),
    }


def run_throughput(client: LocalClient, repeats: int, max_tokens: int, tqdm) -> dict:
    prompt = '请解释在 Agent 系统中工具调用与普通对话的区别，并列举两个实际例子。'
    msgs = [{'role': 'system', 'content': '用中文完整回答，必须符合事实。'}, {'role': 'user', 'content': prompt}]
    samples = []
    with tqdm(total=repeats+1, desc='2-1 流式吞吐 / 首 Token 延迟', unit='请求') as bar:
        warmup = client.stream_chat(msgs, None, max_tokens)
        bar.update(1)
        for i in range(repeats):
            r = client.stream_chat(msgs, None, max_tokens)
            samples.append({'repeat': i+1, **r})
            bar.update(1)
    rates = [i['tokens_per_second'] for i in samples if i['tokens_per_second'] is not None]
    ttfts = [i['ttft_s'] for i in samples if i['ttft_s'] is not None]
    return {'warmup': warmup, 'samples': samples, 'request_messages': msgs,
            'summary': {'repeats': repeats, 'valid_rates': len(rates),
                        'mean_decode_tokens_per_s': statistics.fmean(rates) if rates else None,
                        'median_decode_tokens_per_s': statistics.median(rates) if rates else None,
                        'mean_ttft_s': statistics.fmean(ttfts) if ttfts else None,
                        'exceeded_100_tps_on_this_host': statistics.fmean(rates) > 100 if rates else None,
                        'method': samples[0]['throughput_method'] if samples else None}}


def run_batching(client: LocalClient, levels: list[int], max_tokens: int, tqdm) -> dict:
    results = []
    with tqdm(total=sum(levels), desc='2-1 不同并发度批处理', unit='请求') as bar:
        for concurrency in levels:
            start = time.perf_counter()
            def one(i: int):
                messages = [{'role': 'system', 'content': '请用中文完整回答下述问题。'},
                            {'role': 'user', 'content': f'编号 {i+1}：请简述 Transformer 中注意力机制的作用。'}]
                response = client.stream_chat(messages, None, max_tokens)
                response['request_messages'] = messages
                return response
            requests_results = [None] * concurrency
            errors = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(one, i): i for i in range(concurrency)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        requests_results[idx] = fut.result()
                    except Exception as exc:
                        errors.append({'index': idx, 'error': f'{type(exc).__name__}: {exc}'})
                    bar.update(1)
            wall = time.perf_counter()-start
            counts = [x['usage'].get('completion_tokens') for x in requests_results if x and x.get('usage')]
            valid_tokens = len(counts) == concurrency and all(isinstance(x, int) for x in counts)
            aggregate = sum(counts)/wall if valid_tokens and wall > 0 else None
            results.append({'concurrency': concurrency, 'wall_s': wall,
                            'aggregate_generated_tokens': sum(counts) if valid_tokens else None,
                            'aggregate_decode_tokens_per_s': aggregate, 'requests': requests_results,
                            'errors': errors, 'passed': not errors and valid_tokens})
    return {'levels': results, 'passed': bool(results) and all(r['passed'] for r in results),
            'note': '真实并发请求产生的聚合吞吐；是否有服务端 continuous batching 取决于服务配置，不能只凭吞吐反推内部调度。'}


def run_cache(client: LocalClient, prefix_tokens: int, warmups: int, repeats: int, tqdm) -> dict:
    filler = 'Keep this stable operating-manual sentence unchanged. '
    system = '# Stable operating manual\n' + filler * max(1, int(prefix_tokens*4 / len(filler)))
    question = 'Reply with only the word READY.'
    def request(system_prompt: str) -> dict:
        m = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': question}]
        r = client.stream_chat(m, None, 16)
        return {'ttft_s': r['ttft_s'], 'wall_s': r['wall_s'], 'usage': r['usage'],
                'model_response': r['response'], 'system_sha256': sha256_text(system_prompt),
                'request_messages': m, 'raw': r}
    warmed, pairs = [], []
    with tqdm(total=warmups+2*repeats, desc='2-1 前缀缓存命中/未命中', unit='请求') as bar:
        for _ in range(warmups):
            warmed.append(request(system))
            bar.update(1)
        for i in range(repeats):
            hit = request(system); bar.update(1)
            marker = f'M{i:07d}'
            changed = marker + system[len(marker):]
            miss = request(changed); bar.update(1)
            pairs.append({'index': i+1, 'hit': hit, 'miss': miss,
                          'same_character_length': len(system) == len(changed)})
    hits = [x['hit']['ttft_s'] for x in pairs if x['hit']['ttft_s'] is not None]
    misses = [x['miss']['ttft_s'] for x in pairs if x['miss']['ttft_s'] is not None]
    prompt_equal = all(x['same_character_length'] for x in pairs)
    token_pair_sizes = [
        {'hit': (x['hit'].get('usage') or {}).get('prompt_tokens'),
         'miss': (x['miss'].get('usage') or {}).get('prompt_tokens')}
        for x in pairs
    ]
    return {'stable_system_sha256': sha256_text(system), 'prefix_char_count': len(system),
            'approx_prefix_tokens_requested': prefix_tokens,
            'warmups': warmed, 'pairs': pairs,
            'summary': {'matched_pairs': len(pairs), 'same_length': prompt_equal,
                        'hit_mean_ttft_s': statistics.fmean(hits) if hits else None,
                        'miss_mean_ttft_s': statistics.fmean(misses) if misses else None,
                        'hit_faster_pairs': sum(1 for x in pairs if x['hit']['ttft_s'] is not None and
                                                x['miss']['ttft_s'] is not None and
                                                x['hit']['ttft_s'] < x['miss']['ttft_s']),
                        'prompt_token_pair_sizes': token_pair_sizes,
                        'note': '没有服务端 cache-hit telemetry 时，TTFT 差异只能作为间接证据，不能证明实际命中。'}}


def make_plots(result: dict, folder: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[提示] 未安装 matplotlib，跳过可视化')
        return
    cache = result.get('cache')
    if cache and cache.get('pairs'):
        pairs = cache['pairs']
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [p['index'] for p in pairs]
        ax.plot(xs, [p['hit']['ttft_s'] for p in pairs], marker='o', label='Stable prefix')
        ax.plot(xs, [p['miss']['ttft_s'] for p in pairs], marker='s', label='Changed prefix')
        ax.set(title='TTFT: stable vs mutated system prefix', xlabel='Matched pair', ylabel='TTFT (s)')
        ax.set_xticks(xs); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(folder / 'cache_ttft.png', dpi=180); plt.close(fig)
    batching = result.get('batching')
    if batching and batching.get('levels'):
        levels = batching['levels']
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [x['concurrency'] for x in levels]
        ys = [x['aggregate_decode_tokens_per_s'] or 0 for x in levels]
        ax.bar(range(len(xs)), ys)
        ax.set_xticks(range(len(xs)), [str(x) for x in xs]); ax.set_xlabel('Concurrent requests')
        ax.set_ylabel('Aggregate output tokens/s'); ax.set_title('Measured batching throughput')
        fig.tight_layout(); fig.savefig(folder / 'batching_throughput.png', dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description='完整实验 2-1：本地工具调用、原始 Token、吞吐、并发和 KV Cache')
    parser.add_argument('--backend', choices=['vllm', 'ollama'], default=None)
    parser.add_argument('--base-url', help='本地服务地址：vllm http://127.0.0.1:8000/v1; ollama http://127.0.0.1:11434')
    parser.add_argument('--model', help='本地服务端对外暴露的模型名')
    parser.add_argument('--local-model-dir', help='本地 Qwen3-0.6B 权重目录')
    parser.add_argument('--scenario', choices=['all', 'tools', 'throughput', 'batching', 'cache', 'raw'], default='all')
    parser.add_argument('--raw-engine', choices=['auto', 'service', 'vllm', 'ollama', 'skip'], default='auto',
                        help='两种后端都使用原始生成 API，不额外加载第二份模型到内存')
    parser.add_argument('--raw-device', choices=['cpu', 'cuda'], default='cpu',
                        help=argparse.SUPPRESS)  # 兼容旧版命令；新的 raw 走服务端，无须加载第二个模型
    parser.add_argument('--raw-max-tokens', type=int, default=512)
    parser.add_argument('--max-turns', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--cache-prefix-tokens', type=int, default=4096)
    parser.add_argument('--concurrency', default='1,2,4,8')
    parser.add_argument('--output-dir', help='默认与当前脚本相同目录下的 results/')
    parser.add_argument('--quiet-stream', action='store_true', help='只记录流式片段，不逐块打印模型响应')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.cache_prefix_tokens < 256 or args.max_tokens < 32 or args.raw_max_tokens < 32 or args.max_turns < 1:
        parser.error('repeats>=1, warmups>=0, cache-prefix-tokens>=256, max-tokens>=32, raw-max-tokens>=32, max-turns>=1')
    try:
        levels = [int(x.strip()) for x in args.concurrency.split(',')]
        if not levels or any(x < 1 or x > 32 for x in levels):
            raise ValueError()
    except ValueError:
        parser.error('--concurrency 格式例如 1,2,4,8；并发度限 1..32')
    cfg = optional_config()
    args.backend = args.backend or cfg.get('backend') or 'vllm'
    model_dir = Path(args.local_model_dir or cfg.get('local_model_dir') or DEFAULT_MODEL_DIR).expanduser().resolve()
    base_url = args.base_url or cfg.get('base_url') or ('http://127.0.0.1:8000/v1' if args.backend == 'vllm' else 'http://127.0.0.1:11434')
    assert_local_url(base_url)
    model = args.model or cfg.get('model')
    raw_engine = args.raw_engine
    if raw_engine == 'auto':
        raw_engine = 'service'
    output_parent = Path(args.output_dir or cfg.get('output_dir') or (Path(__file__).resolve().parent / 'results'))
    if not output_parent.is_absolute():
        output_parent = root_dir() / output_parent
    if args.dry_run:
        print(json.dumps({'experiment': '2-1', 'backend': args.backend, 'api': base_url,
                          'model': model or '服务端匹配模型名', 'model_dir': str(model_dir),
                          'local_model_dir_exists': model_dir.is_dir(),
                          'local_config_exists': (model_dir / 'config.json').is_file(),
                          'weights_found': bool(list(model_dir.glob('*.safetensors'))),
                          'run_scenarios': ['tools', 'raw', 'throughput', 'batching', 'cache'] if args.scenario == 'all' else [args.scenario],
                          'raw_engine': raw_engine, 'raw_device': 'not_applicable_server_inference',
                          'model_download': False, 'remote_llm_api': False,
                          'vllm_start': launch_command(model_dir),
                          'output_dir': str(output_parent)}, ensure_ascii=False, indent=2))
        return 0
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit('缺少运行依赖：python -m pip install requests tqdm matplotlib') from exc
    client = LocalClient(args.backend, base_url, model or '', timeout=360)
    try:
        available = client.models()
    except Exception as exc:
        print(f'[错误] 连接本地服务失败：{exc}\n请先启动：{launch_command(model_dir)}', file=sys.stderr)
        return 2
    if model and model not in available:
        print(f'[错误] 指定模型 {model} 不在本地服务列表：{available}', file=sys.stderr)
        return 2
    if not model:
        favorites = [DEFAULT_SERVED_NAME, str(model_dir), model_dir.name, 'Qwen/Qwen3-0.6B'] if args.backend == 'vllm' else ['qwen3:0.6b']
        model = next((x for x in favorites if x in available), None)
        if model is None and len(available) == 1:
            model = available[0]
        if not model:
            print(f'[错误] 未匹配到本地模型，服务端提供：{available}，请使用 --model', file=sys.stderr)
            return 2
    client.model = model
    server_identity = read_local_server_identity(client, model_dir)
    print(f'[本地模型身份] 已核验：{server_identity["identity_verified"]} | '
          f'摘要：{(server_identity.get("model_digest") or "未返回")[:28]}')
    folder = output_parent / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    required = ['tools', 'raw', 'throughput', 'batching', 'cache'] if args.scenario == 'all' else [args.scenario]
    report = {'experiment_id': '2-1', 'started_at': datetime.now().isoformat(),
              'host': platform.platform(), 'backend': args.backend, 'model_served_name': model,
              'model_dir_requested': str(model_dir), 'base_url': base_url,
              'model_source_caveat': '必须以本地服务返回的模型 root/digest 核对；不能只凭 model ID 判断权重身份。',
              'server_identity': server_identity,
              'official_protocol_equivalent': args.backend == 'ollama' and raw_engine in ('service', 'ollama'),
              'config': {'args': vars(args), 'raw_engine_resolved': raw_engine,
                         'requested_scenarios': required, 'concurrency': levels}, 'scenarios': {}}
    print(f'[实验 2-1] 模型：{model} / 后端：{args.backend} / 输出：{folder}')
    for scenario in required:
        print(f'\n--- 2-1: {scenario} ---')
        try:
            if scenario == 'tools':
                data = tool_roundtrip(client, args.max_turns, args.max_tokens, tqdm, show_stream=not args.quiet_stream)
                ok = data['passed']
            elif scenario == 'raw':
                data = run_raw_experiment(client, model_dir, raw_engine, args.raw_device,
                                          args.raw_max_tokens, tqdm, show_stream=not args.quiet_stream)
                ok = bool(data.get('passed'))
            elif scenario == 'throughput':
                data = run_throughput(client, args.repeats, min(256, args.max_tokens), tqdm)
                ok = data['summary']['valid_rates'] == args.repeats
            elif scenario == 'batching':
                data = run_batching(client, levels, min(128, args.max_tokens), tqdm)
                ok = data['passed']
            elif scenario == 'cache':
                data = run_cache(client, args.cache_prefix_tokens, args.warmups, args.repeats, tqdm)
                ok = (data['summary']['matched_pairs'] == args.repeats and
                      data['summary']['same_length'] and
                      all(pair['hit']['ttft_s'] is not None and pair['miss']['ttft_s'] is not None
                          for pair in data['pairs']))
            else:
                raise AssertionError(scenario)
            report['scenarios'][scenario] = {'status': 'passed' if ok else 'incomplete', 'data': data}
            print(f'[{scenario}] ' + ('完成' if ok else '未满足全部条件'))
        except Exception as exc:
            report['scenarios'][scenario] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
            print(f'[{scenario}] 失败：{type(exc).__name__}: {exc}', file=sys.stderr)
        save_json(folder / f'{scenario}.json', report['scenarios'][scenario])
    make_plots({k: (v['data'] if v['status'] != 'error' else None)
                for k, v in report['scenarios'].items()}, folder)
    report['finished_at'] = datetime.now().isoformat()
    report['gates'] = {name: report['scenarios'].get(name, {}).get('status') == 'passed'
                       for name in required}
    report['status'] = 'complete' if all(report['gates'].values()) else 'partial_or_failed'
    report['official_exact_mode'] = (
        args.scenario == 'all' and args.backend == 'ollama' and
        model == 'qwen3:0.6b' and server_identity['identity_verified'] and
        raw_engine in ('service', 'ollama') and args.repeats == 5 and
        args.warmups == 2 and args.cache_prefix_tokens == 4096 and
        args.raw_max_tokens == 512 and args.max_tokens == 512
    )
    report['official_protocol_match'] = bool(report['official_exact_mode'])
    # 本脚本与原仓库 run_experiment.py 的冻结系统消息/工具实现并非字节完全相同。
    # 不冒充原仓库 acceptance 运行身份；这不影响章节核心机制的完整独立复现。
    report['official_exact_complete'] = False
    report['official_exact_note'] = (
        '完整覆盖正文机制及配套 benchmark；但本单文件版本不等于官方冻结版本的 '
        'run_experiment.py/experiment_protocol.json 字节级复现，因此官方严格验收为 False。'
    )
    if args.scenario == 'all' and args.backend == 'vllm':
        report['protocol_note'] = ('vLLM 的 raw 流使用 /v1/completions + 原始 Chat Template；无需额外加载模型。'
                                   '可完成正文机制，但与官方固定的 Ollama raw 引擎身份不完全相同。')
    save_json(folder / 'experiment_results.json', report)
    files = [p for p in folder.iterdir() if p.is_file()]
    manifest = {'experiment': '2-1', 'status': report['status'], 'gates': report['gates'],
                'files': {p.name: {'size_bytes': p.stat().st_size,
                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
    save_json(folder / 'manifest.json', manifest)
    print('\n[结果汇总] ' + json.dumps({'status': report['status'], 'gates': report['gates']}, ensure_ascii=False))
    print(f'[输出] {folder}')
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
