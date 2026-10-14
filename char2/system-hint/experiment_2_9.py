#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 2-9: matched, real-API Agent status-bar ablations.

Closely follows bojieli/ai-agent-book chapter2/system-hint's frozen matched task
mechanism (upstream executable happens to be called run_experiment_2_8.py).
Standalone, limited to isolated file tools; supports DashScope/Qwen.
No actual OS package installation is ever performed.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Embedded support functions: this experiment file is independently runnable.
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_environment() -> dict:
    root = project_root()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
    except ImportError:
        if (root / '.env').is_file():
            for line in (root / '.env').read_text(encoding='utf-8-sig').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip().removeprefix('export ').strip()
                v = v.strip().strip('\'"')
                if k.isidentifier():
                    os.environ.setdefault(k, v)
    try:
        import yaml
        content = root / 'config.yaml'
        data = yaml.safe_load(content.read_text(encoding='utf-8-sig')) if content.is_file() else {}
        return data if isinstance(data, dict) else {}
    except (ImportError, OSError, ValueError):
        return {}


def resolve_provider(provider: str, model: str | None, base_url: str | None) -> dict:
    cfg = load_environment()
    m = cfg.get('model', {})
    from_yaml = m.get('name') if isinstance(m, dict) else None
    if provider == 'dashscope':
        return {'provider': provider, 'model': model or from_yaml or os.getenv('MODEL_NAME') or 'qwen3.7-flash',
                'base_url': base_url or os.getenv('QWEN_API_BASE_URL') or os.getenv('DASHSCOPE_BASE_URL') or
                            'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': os.getenv('DASHSCOPE_API_KEY')}
    if provider == 'moonshot':
        return {'provider': provider, 'model': model or 'kimi-k3',
                'base_url': base_url or os.getenv('MOONSHOT_BASE_URL') or 'https://api.moonshot.cn/v1',
                'api_key': os.getenv('MOONSHOT_API_KEY') or os.getenv('KIMI_API_KEY')}
    if provider == 'local':
        return {'provider': provider, 'model': model or 'qwen3_0.6b',
                'base_url': base_url or 'http://127.0.0.1:8000/v1', 'api_key': 'EMPTY'}
    raise ValueError('invalid provider')


def make_client(cfg: dict, timeout: float = 120):
    if not cfg.get('api_key'):
        raise ValueError('缺少 API Key，请检查项目根目录的 .env')
    from openai import OpenAI
    return OpenAI(api_key=cfg['api_key'], base_url=cfg['base_url'], timeout=timeout, max_retries=1)


def stamp() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_suffix(path.suffix + '.tmp')
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    t.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_json(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                     default=str).encode('utf-8')).hexdigest()


def configured_secrets(cfg: dict) -> list[str]:
    names = ('DASHSCOPE_API_KEY', 'KIMI_API_KEY', 'MOONSHOT_API_KEY')
    return [x for x in [cfg.get('api_key'), *(os.getenv(n) for n in names)] if x and x != 'EMPTY']


def scan_credentials(path: Path, cfg: dict) -> list[str]:
    secrets = configured_secrets(cfg)
    findings = []
    for f in sorted(path.rglob('*')):
        if not f.is_file() or f.suffix not in ('.json', '.jsonl', '.txt', '.md'):
            continue
        body = f.read_text(encoding='utf-8', errors='replace')
        if any(secret in body for secret in secrets):
            findings.append(str(f.relative_to(path)))
    return findings


def openai_chat(client, cfg: dict, messages: list, tools: list | None, temperature: float,
                max_tokens: int, timeout: float | None = None, tool_choice: Any = None) -> tuple[dict, dict]:
    kwargs = {'model': cfg['model'], 'messages': messages, 'temperature': temperature,
              'max_tokens': max_tokens}
    if tools:
        kwargs['tools'] = tools
        kwargs['tool_choice'] = tool_choice if tool_choice is not None else 'auto'
    if cfg['provider'] == 'dashscope':
        kwargs['extra_body'] = {'enable_thinking': False}
    if timeout is not None:
        kwargs['timeout'] = timeout
    started = time.perf_counter()
    raw_request = json.loads(json.dumps(kwargs, ensure_ascii=False))
    # No client credentials, or HTTP headers, are written to the logs.
    try:
        response = client.chat.completions.create(**kwargs)
        data = response.model_dump(mode='json')
        receipt = {'request': raw_request, 'response': data,
                   'elapsed_s': round(time.perf_counter() - started, 4)}
        return data, receipt
    except Exception as exc:
        receipt = {'request': raw_request, 'elapsed_s': round(time.perf_counter() - started, 4),
                   'error': {'type': type(exc).__name__, 'message': str(exc)}}
        return {}, receipt


def receipt_ok(receipt: dict) -> bool:
    response = receipt.get('response') or {}
    return bool(response.get('id') and isinstance(response.get('choices'), list)
                and response.get('choices') and (response.get('usage') or {}).get('total_tokens') is not None)


def usage_summary(receipts: list[dict]) -> dict:
    sums = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cached_tokens_reported': 0}
    cached_seen = False
    for rec in receipts:
        usage = (rec.get('response') or {}).get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            sums[key] += int(usage.get(key) or 0)
        detail = usage.get('prompt_tokens_details') or {}
        if detail.get('cached_tokens') is not None:
            cached_seen = True
            sums['cached_tokens_reported'] += int(detail.get('cached_tokens') or 0)
    sums['cached_tokens_reported'] = sums['cached_tokens_reported'] if cached_seen else None
    sums['api_calls'] = len(receipts)
    sums['valid_api_calls'] = sum(receipt_ok(x) for x in receipts)
    return sums


def local_tiktoken_count(text: str) -> tuple[int, str]:
    try:
        import tiktoken
        return len(tiktoken.get_encoding('cl100k_base').encode(text)), 'cl100k_base_estimate_not_qwen'
    except ImportError:
        return max(1, len(text.encode('utf-8')) // 3), 'utf8_bytes_div3_estimate'


def token_estimate(messages: list[dict]) -> tuple[int, str]:
    text = json.dumps(messages, ensure_ascii=False, separators=(',', ':'))
    n, method = local_tiktoken_count(text)
    return n + len(messages) * 4, method


def manifest_for(path: Path, cfg: dict, experiment: str, complete: bool) -> dict:
    artifacts = {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json' and not p.name.endswith('.tmp')}
    findings = scan_credentials(path, cfg)
    out = {'experiment': experiment, 'status': 'complete' if complete and not findings else 'partial',
           'credential_scan_passed': not bool(findings), 'credential_findings': findings,
           'sha256_files': artifacts}
    atomic_json(path / 'manifest.json', out)
    return out


CONDITIONS = {
    'disabled': [], 'timestamps_raw': ['timestamps'],
    'timestamps_guided': ['timestamps', 'timestamp_guidance'],
    'tool_counter': ['tool_counter'], 'todo_list': ['todo_list'],
    'detailed_errors': ['detailed_errors'], 'system_state': ['system_state'],
    'combined': ['timestamps', 'timestamp_guidance', 'tool_counter', 'todo_list',
                 'detailed_errors', 'system_state'],
}
CONTRASTS = ['timestamps_raw', 'timestamps_guided', 'tool_counter', 'todo_list',
             'detailed_errors', 'system_state', 'combined']
SUITES = {'timestamps_raw': 'timestamps', 'timestamps_guided': 'timestamps',
          'tool_counter': 'tool_counter', 'todo_list': 'todo_list',
          'detailed_errors': 'detailed_errors', 'system_state': 'system_state',
          'combined': 'combined'}

# The first five fixed matched examples in the upstream experiment_protocol.json,
# transcribed here so no GitHub download is required in WSL.
TS = [
    ('ts-01', {'cedar':'2025-09-13 22:10:00','maple':'2025-09-14 09:05:00'}, 'maple'),
    ('ts-02', {'amber':'2025-09-14 11:45:00','indigo':'2025-09-14 08:20:00'}, 'amber'),
    ('ts-03', {'north':'2025-09-12 17:30:00','south':'2025-09-13 06:15:00'}, 'south'),
    ('ts-04', {'lima':'2025-09-14 14:00:01','oslo':'2025-09-14 14:00:00'}, 'lima'),
    ('ts-05', {'quartz':'2025-09-11 23:59:59','river':'2025-09-12 00:00:01'}, 'river')]
CASES = {
    'timestamps': [dict(id=i, records=rs, expected=ex) for i,rs,ex in TS],
    'tool_counter': [dict(id=f'ctr-0{i+1}',primary=f'gateway-{x}',fallback=f'mirror-{x}')
                     for i,x in enumerate('abcde')],
    'todo_list': [dict(id=f'todo-0{i+1}', token=t, artifacts=a) for i,(t,a) in enumerate([
        ('ALPHA-417',['inventory.txt','decision.txt','audit.txt','summary.txt']),
        ('BRAVO-528',['inputs.txt','analysis.txt','checks.txt','delivery.txt']),
        ('CHARLIE-639',['scope.txt','plan.txt','verification.txt','result.txt']),
        ('DELTA-740',['sources.txt','matrix.txt','review.txt','report.txt']),
        ('ECHO-851',['request.txt','work.txt','quality.txt','handoff.txt'])])],
    'detailed_errors': [dict(id=f'err-0{i+1}',requested=req,actual=actual,token=t)
                        for i,(req,actual,t) in enumerate([
        ('invoice.txt','invoice_2025.txt','INV-2041'),
        ('policy.md','policy_final.md','POL-3152'),
        ('metrics.csv','metrics_v2.csv','MET-4263'),
        ('brief.txt','brief_revised.txt','BRF-5374'),
        ('manifest.json','manifest_current.json','MAN-6485')])],
    'system_state': [dict(id=f'state-0{i+1}',os=o,shell=s,python=p,cwd=d,manager=m,package=pkg)
                     for i,(o,s,p,d,m,pkg) in enumerate([
        ('Linux','bash','3.11.9','workspace/alpha','apt','jq'),
        ('Darwin','zsh','3.12.4','workspace/beta','brew','ripgrep'),
        ('Windows','PowerShell','3.11.8','workspace/gamma','winget','Git.Git'),
        ('Linux','fish','3.10.14','workspace/delta','apt','curl'),
        ('Darwin','zsh','3.13.0','workspace/epsilon','brew','tree')])],
    'combined': []
}
_combined = [
    ('all-01',{'oak':'2025-09-13 08:00:00','pine':'2025-09-14 08:00:00'},'pine',
     'core-a','backup-a','config.txt','config_live.txt','CFG-711','Linux','bash','3.11.9','workspace/one','apt','jq',['observe.txt','recover.txt','deliver.txt']),
    ('all-02',{'red':'2025-09-15 10:01:00','blue':'2025-09-15 10:00:00'},'red',
     'core-b','backup-b','runbook.md','runbook_v3.md','RUN-822','Darwin','zsh','3.12.4','workspace/two','brew','ripgrep',['timeline.txt','diagnosis.txt','handoff.txt']),
    ('all-03',{'east':'2025-09-16 12:00:00','west':'2025-09-16 12:00:02'},'west',
     'core-c','backup-c','ledger.csv','ledger_final.csv','LED-933','Windows','PowerShell','3.11.8','workspace/three','winget','Git.Git',['facts.txt','remediation.txt','evidence.txt']),
    ('all-04',{'gold':'2025-09-17 07:30:00','silver':'2025-09-17 07:29:59'},'gold',
     'core-d','backup-d','notes.txt','notes_latest.txt','NOT-144','Linux','fish','3.10.14','workspace/four','apt','curl',['inputs.txt','fallback.txt','final.txt']),
    ('all-05',{'sun':'2025-09-18 18:45:00','moon':'2025-09-18 19:15:00'},'moon',
     'core-e','backup-e','status.json','status_current.json','STA-255','Darwin','zsh','3.13.0','workspace/five','brew','tree',['state.txt','actions.txt','closure.txt'])
]
for row in _combined:
    (id_, records, expected_record, primary, fallback, requested, actual, token, os_, shell,
     python_, cwd, manager, package, artifacts) = row
    CASES['combined'].append(dict(id=id_,records=records,expected_record=expected_record,
         primary=primary,fallback=fallback,requested=requested,actual=actual,token=token,
         os=os_,shell=shell,python=python_,cwd=cwd,manager=manager,package=package,artifacts=artifacts))


def case_prompt(suite: str, case: dict) -> str:
    if suite == 'timestamps':
        names = list(case['records'])
        return f'Read both records {names[0]} and {names[1]}. Select the record with the later recorded timestamp and submit its name. Do not guess.'
    if suite == 'tool_counter':
        return f"Retrieve the payload for primary resource {case['primary']}. If unavailable, use fallback {case['fallback']}. Submit the resource that supplied the payload."
    if suite == 'todo_list':
        return f"Create exactly these artifacts: {', '.join(case['artifacts'])}. Every artifact must contain exactly the token {case['token']}. Verify then submit the complete artifact list."
    if suite == 'detailed_errors':
        return f"Read {case['requested']}. It may have been renamed. Recover the actual document without inventing its contents, then submit filename and exact token."
    if suite == 'system_state':
        return f"Install package {case['package']} in the simulated host with the OS-native package manager. Inspect if needed. Submit manager and package."
    names = list(case['records'])
    return (f"Complete ALL five incident components. (1) Read records {names[0]} and {names[1]} and select later. "
            f"(2) Probe primary {case['primary']} or fallback {case['fallback']}. "
            f"(3) Write artifacts {', '.join(case['artifacts'])} with exact token {case['token']}. "
            f"(4) Recover renamed document {case['requested']} and its exact token. "
            f"(5) Install package {case['package']} with the simulated host's native package manager. "
            "Use evidence tools and submit one combined result.")


def setup_sandbox(sandbox: Path, suite: str, case: dict) -> dict:
    if sandbox.exists():
        raise FileExistsError('sandbox already exists; use a new output directory or validated --resume')
    sandbox.mkdir(parents=True)
    state = {'suite': suite, 'case_id': case['id']}
    if suite in ('timestamps','combined'):
        state['records'] = case['records']
        (sandbox/'records').mkdir()
        for name in case['records']:
            (sandbox/'records'/f'{name}.txt').write_text(f'payload-for-{name}\n', encoding='utf-8')
    if suite in ('tool_counter','combined'):
        state['primary'], state['fallback'] = case['primary'],case['fallback']
        (sandbox/'resources').mkdir()
        (sandbox/'resources'/f"{case['fallback']}.txt").write_text(f"verified-payload-{case['id']}\n",encoding='utf-8')
    if suite in ('todo_list','combined'):
        state['token'],state['artifacts'] = case['token'],case['artifacts']
        (sandbox/'artifacts').mkdir()
    if suite in ('detailed_errors','combined'):
        state['requested'],state['actual'] = case['requested'],case['actual']
        (sandbox/'documents').mkdir()
        (sandbox/'documents'/case['actual']).write_text(case['token']+'\n', encoding='utf-8')
    if suite in ('system_state','combined'):
        state['system'] = {x:case[x] for x in ('os','shell','python','cwd','manager','package')}
    atomic_json(sandbox/'initial_state.json',state)
    return state


def schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {'type':'function','function':{'name':name,'description':description,
            'parameters':{'type':'object','properties':properties,'required':required,
                          'additionalProperties':False}}}


def tools_for(suite: str, features: set[str]) -> list[dict]:
    ss = lambda info: {'type':'string','description':info}
    result = []
    if suite in ('timestamps','combined'):
        result.append(schema('read_record','Read a record by name',{'name':ss('Record name')},['name']))
    if suite in ('tool_counter','combined'):
        result.append(schema('probe_resource','Read a primary or fallback resource',{'name':ss('Resource name')},['name']))
    if suite in ('todo_list','combined'):
        result.append(schema('write_artifact','Write artifact in isolated sandbox',
                             {'file':ss('Required filename'),'content':ss('Exact content')},['file','content']))
        if 'todo_list' in features:
            result.append(schema('rewrite_todo_list','Create checklist',
                                 {'items':{'type':'array','items':{'type':'string'}}},['items']))
            result.append(schema('update_todo_status','Update checklist item',
                                 {'item':ss('Item'),'status':{'type':'string','enum':['pending','in_progress','completed','cancelled']}},
                                 ['item','status']))
    if suite in ('detailed_errors','combined'):
        result.extend([schema('read_document','Read named document',{'file':ss('Document filename')},['file']),
                       schema('list_documents','List available documents',{},[])])
    if suite in ('system_state','combined'):
        result.extend([schema('inspect_system','Read simulated host state',{},[]),
                       schema('install_package','Simulate native package installation (no host changes)',
                              {'manager':ss('Package manager'),'package':ss('Package name')},['manager','package'])])
    params = {
      'timestamps':({'selected_record':ss('Later record')},['selected_record']),
      'tool_counter':({'fallback':ss('Resource that supplied payload')},['fallback']),
      'todo_list':({'artifacts':{'type':'array','items':{'type':'string'}}},['artifacts']),
      'detailed_errors':({'document':ss('Renamed file'),'token':ss('Recovered token')},['document','token']),
      'system_state':({'manager':ss('Manager used'),'package':ss('Package installed')},['manager','package'])
    }
    if suite == 'combined':
        p = {k:v for s in ('timestamps','tool_counter','todo_list','detailed_errors','system_state') for k,v in params[s][0].items()}
        need = list(p)
    else:
        p,need = params[suite]
    result.append(schema('submit_result','Submit final result after checking evidence',p,need))
    return result


def status_hint(features: set[str], state: dict, counts: dict, todos: dict) -> str:
    parts = []
    if 'timestamp_guidance' in features:
        parts.append('TIME GUIDANCE: Compare the timestamps in tool returns; do not guess which record is later.')
    if 'tool_counter' in features:
        parts.append('TOOL COUNTS: ' + (', '.join(f'{k}={v}' for k,v in sorted(counts.items())) or 'none') +
                     '. Diagnose persistent failures and use the named fallback.')
    if 'todo_list' in features:
        parts.append('TODO LIST: '+('; '.join(f'{k}:{v}' for k,v in todos.items()) or 'not created')+
                     '. Verify each required artifact.')
    if 'system_state' in features:
        parts.append('SYSTEM STATE: ' + ', '.join(f'{k}={v}' for k,v in state.get('system',{}).items() if k!='package'))
    return '<agent_status>\n'+'\n'.join(parts)+'\n</agent_status>' if parts else ''


def tool_execute(sandbox: Path, state: dict, features: set[str], name: str,
                 args: dict, counts: dict, todos: dict) -> tuple[str,bool]:
    counts[name] = counts.get(name,0)+1
    ok = True
    if name == 'read_record':
        record = str(args.get('name',''))
        if record not in state.get('records',{}):
            ok, content=False,'Error: record not found'
        else:
            content = (sandbox/'records'/f'{record}.txt').read_text().strip()
            if 'timestamps' in features:
                content = '['+state['records'][record]+'] '+content
    elif name == 'probe_resource':
        res = str(args.get('name',''))
        if res==state.get('primary'):
            ok,content=False,'Error: primary resource unavailable'
        elif res==state.get('fallback'):
            content=(sandbox/'resources'/f'{res}.txt').read_text().strip()
        else:
            ok,content=False,'Error: unknown resource'
    elif name == 'write_artifact':
        filename=str(args.get('file',''))
        if filename not in state.get('artifacts',[]):
            ok,content=False,'Error: filename outside allowed artifact set'
        else:
            (sandbox/'artifacts'/filename).write_text(str(args.get('content','')),encoding='utf-8')
            content=f'Wrote artifacts/{filename}'
    elif name == 'rewrite_todo_list' and 'todo_list' in features:
        items=args.get('items',[])
        if not isinstance(items,list):
            ok,content=False,'Error: items must be an array'
        else:
            todos.clear()
            for item in items[:30]: todos[str(item)]='pending'
            content=f'TODO list contains {len(todos)} items'
    elif name == 'update_todo_status' and 'todo_list' in features:
        item,status=str(args.get('item','')),str(args.get('status',''))
        if item not in todos or status not in ('pending','in_progress','completed','cancelled'):
            ok,content=False,'Error: invalid TODO update'
        else:
            todos[item]=status
            content=f'Updated {item}={status}'
    elif name == 'read_document':
        filename=str(args.get('file',''))
        if filename not in (state.get('requested'),state.get('actual')) or filename != state.get('actual'):
            ok=False
            if 'detailed_errors' in features:
                content=json.dumps({'error_type':'FileNotFoundError','file':filename,
                                    'arguments':args,'stack':'sandbox.read_document -> open',
                                    'available_documents':[state.get('actual')],
                                    'suggestion':'list_documents then read renamed file'},ensure_ascii=False)
            else:
                content='Error: document not found'
        else:
            content=(sandbox/'documents'/filename).read_text().strip()
    elif name == 'list_documents':
        if counts.get('read_document',0) == 0:
            ok,content=False,'Error: first attempt the user-requested document path'
        else:
            content=json.dumps([state.get('actual')]) if state.get('actual') else '[]'
    elif name == 'inspect_system':
        content=json.dumps(state.get('system',{}),sort_keys=True)
    elif name == 'install_package':
        expected=state.get('system',{})
        if args.get('manager') == expected.get('manager') and args.get('package') == expected.get('package'):
            atomic_json(sandbox/'install_action.json',{'manager':args['manager'],'package':args['package']})
            content='Simulated install succeeded'
        else:
            ok,content=False,'Error: wrong native manager or package'
    elif name == 'submit_result':
        content='Submission recorded'
    else:
        ok,content=False,'Error: unknown or disallowed tool'
    if 'tool_counter' in features:
        content += f"\nTool call #{counts[name]} for '{name}'."
    return content,ok


def objective_scores(suite: str, case: dict, events: list[dict], sandbox: Path) -> dict:
    submitted=[e.get('args',{}) for e in events if e['name']=='submit_result']
    answer=submitted[-1] if submitted else {}
    scores={}
    if suite in ('timestamps','combined'):
        records={e.get('args',{}).get('name') for e in events if e['name']=='read_record' and e['ok']}
        scores['timestamps']=(set(case['records']).issubset(records)
                              and answer.get('selected_record')==case.get('expected',case.get('expected_record')))
    if suite in ('tool_counter','combined'):
        probes=sum(e['name']=='probe_resource' and e.get('args',{}).get('name')==case['primary'] for e in events)
        fallback=any(e['name']=='probe_resource' and e['ok'] and e.get('args',{}).get('name')==case['fallback'] for e in events)
        scores['tool_counter']=bool(fallback and probes<=3 and answer.get('fallback')==case['fallback'])
    if suite in ('todo_list','combined'):
        correct=all((sandbox/'artifacts'/file).is_file() and
                    (sandbox/'artifacts'/file).read_text(encoding='utf-8').strip()==case['token']
                    for file in case['artifacts'])
        submitted_artifacts=answer.get('artifacts',[])
        scores['todo_list']=bool(correct and isinstance(submitted_artifacts,list)
                                 and set(submitted_artifacts)==set(case['artifacts']))
    if suite in ('detailed_errors','combined'):
        tried=any(e['name']=='read_document' and e.get('args',{}).get('file')==case['requested'] and not e['ok'] for e in events)
        read=any(e['name']=='read_document' and e.get('args',{}).get('file')==case['actual'] and e['ok'] for e in events)
        scores['detailed_errors']=bool(tried and read and answer.get('document')==case['actual'] and answer.get('token')==case['token'])
    if suite in ('system_state','combined'):
        installed=any(e['name']=='install_package' and e['ok'] for e in events)
        scores['system_state']=bool(installed and answer.get('manager')==case['manager'] and answer.get('package')==case['package'])
    return scores


def validate_tool_protocol(messages: list[dict]) -> bool:
    pending=[]
    for msg in messages:
        if pending:
            if msg.get('role')!='tool' or msg.get('tool_call_id')!=pending.pop(0):
                return False
        elif msg.get('role')=='assistant' and msg.get('tool_calls'):
            pending=[x['id'] for x in msg['tool_calls']]
    return not pending


def sandbox_inventory(sandbox: Path) -> dict[str,str]:
    return {p.relative_to(sandbox).as_posix():sha256(p)
            for p in sorted(sandbox.rglob('*')) if p.is_file()}


def run_case(client, cfg: dict, args, output: Path, contrast: str, case: dict,
             condition: str, order: list[str], order_position: int, protocol_hash: str) -> dict:
    suite=SUITES[contrast]
    run_id=f'{contrast}__{case["id"]}__{condition}'
    fp=output/'cases'/f'{run_id}.json'
    sandbox=output/'sandboxes'/run_id
    if fp.exists():
        old=json.loads(fp.read_text(encoding='utf-8'))
        if (not args.resume or old.get('protocol_hash')!=protocol_hash or
            not old.get('trace_complete') or old.get('model')!=cfg['model'] or
            old.get('sandbox_inventory')!=sandbox_inventory(sandbox)):
            raise RuntimeError(f'Existing run {fp} cannot be safely reused; use a new --output-dir')
        return old
    state=setup_sandbox(sandbox,suite,case)
    features=set(CONDITIONS[condition])
    user_prompt=case_prompt(suite,case)
    if 'timestamps' in features:
        user_prompt='[2025-09-19 09:00:00] '+user_prompt
    messages=[{'role':'system','content':
        'You are an audited local-sandbox Agent. Only use the supplied simulated tools. '
        'Use actual tool results, never invent environment facts, and call submit_result '
        'once you have sufficient evidence. No real shell or external side effects exist.'},
        {'role':'user','content':user_prompt}]
    counts,todos={},{}
    events,receipts=[],[]
    termination='max_steps'
    for step in range(1,args.max_steps+1):
        request_messages=copy.deepcopy(messages)
        hint=status_hint(features,state,counts,todos)
        if hint:
            request_messages.append({'role':'user','content':hint})  # never persist ephemeral hint
        result,rec=openai_chat(client,cfg,request_messages,tools_for(suite,features),
                               args.temperature,args.max_tokens,timeout=args.timeout)
        receipts.append(rec)
        if not receipt_ok(rec):
            termination='api_error_or_missing_receipt'
            break
        assistant=copy.deepcopy(result['choices'][0]['message'])
        assistant={k:v for k,v in assistant.items() if k in ('role','content','tool_calls','reasoning_content') and v is not None}
        assistant['role']='assistant'
        messages.append(assistant)
        calls=assistant.get('tool_calls') or []
        if not calls:
            messages.append({'role':'user','content':'Submit the task result with the submit_result tool after evidence checks. If impossible, report why.'})
            continue
        submitted=False
        for call in calls:
            name=call.get('function',{}).get('name','')
            raw=call.get('function',{}).get('arguments','{}')
            try:
                arguments=json.loads(raw) if isinstance(raw,str) else raw
                if not isinstance(arguments,dict): raise ValueError('tool arguments must be object')
                outcome,ok=tool_execute(sandbox,state,features,name,arguments,counts,todos)
            except (ValueError,TypeError) as exc:
                arguments={}
                outcome,ok=f'Invalid tool arguments: {exc}',False
            events.append({'step':step,'name':name,'args':arguments,'ok':ok,'result':outcome,
                           'call_id':call.get('id')})
            messages.append({'role':'tool','tool_call_id':call.get('id'),'name':name,'content':outcome})
            if name=='submit_result': submitted=True
        if submitted:
            termination='submitted'
            break
    scores=objective_scores(suite,case,events,sandbox)
    # Completion requires a final submit_result, not merely reaching the step budget.
    # A non-passing objective score is still a valid completed ablation if submitted.
    completed=(bool(receipts) and all(receipt_ok(x) for x in receipts)
               and termination == 'submitted' and validate_tool_protocol(messages))
    payload={'experiment':'2-9','contrast':contrast,'suite':suite,'case_id':case['id'],'condition':condition,
             'features':sorted(features),'order':order,'order_position':order_position,
             'protocol_hash':protocol_hash,'model':cfg['model'],'messages':messages,
             'api_calls':receipts,'tool_events':events,'tool_counter':counts,'todo':todos,
             'objective_scores':scores,'objective_pass':bool(scores) and all(scores.values()),
             'steps':len(receipts),'termination':termination,'usage':usage_summary(receipts),
             'tool_protocol_valid':validate_tool_protocol(messages),
             'sandbox_inventory':sandbox_inventory(sandbox),
             'trace_complete':completed, 'submission_recorded':termination == 'submitted'}
    atomic_json(fp,payload)
    return payload


def compare(rows: list[dict], contrasts: list[str], args) -> dict:
    summaries=[]
    for contrast in contrasts:
        cells=[r for r in rows if r['contrast']==contrast]
        cases=CASES[SUITES[contrast]][:args.cases_per_suite]
        enabled=[next(r for r in cells if r['case_id']==case['id'] and r['condition']==contrast) for case in cases]
        disabled=[next(r for r in cells if r['case_id']==case['id'] and r['condition']=='disabled') for case in cases]
        def metrics(rows):
            return {'pass_count':sum(x['objective_pass'] for x in rows), 'n':len(rows),
                    'mean_steps':round(sum(x['steps'] for x in rows)/len(rows),3),
                    'mean_tools':round(sum(len(x['tool_events']) for x in rows)/len(rows),3),
                    'tool_errors':sum(not e['ok'] for x in rows for e in x['tool_events']),
                    'tokens':sum(x['usage']['total_tokens'] for x in rows),
                    'trace_complete':sum(x['trace_complete'] for x in rows)}
        a,b=metrics(enabled),metrics(disabled)
        summaries.append({'contrast':contrast,'enabled':a,'disabled':b,
                          'delta_pass_rate':round((a['pass_count']-b['pass_count'])/len(cases),3),
                          'qualifier':'observed only; a loss or tie is a valid result'})
    all_receipts=[v for x in rows for v in x['api_calls']]
    return {'experiment':'2-9','status':'complete' if all(x['trace_complete'] for x in rows) else 'partial',
            'provider_model':rows[0]['model'] if rows else None,
            'expected_runs':2*args.cases_per_suite*len(contrasts),'observed_runs':len(rows),
            'actual_api':True,'all_case_traces_complete':all(x['trace_complete'] for x in rows),
            'usage':usage_summary(all_receipts),'contrasts':summaries,
            'source':'Adapted fixed cases from bojieli/ai-agent-book chapter2/system-hint; non-frozen Qwen run',
            'objective_scoring':'tool receipts + isolated filesystem + exact submitted values',
            'historical_claims_reproduced':False}



def render_report(comparison: dict, output: Path) -> None:
    """Produce readable Markdown; plot optional, absence not an experiment failure."""
    lines=['# 实验 2-9：Agent 状态栏对照结果', '',
           f"模型：`{comparison['provider_model']}`；状态：`{comparison['status']}`；运行次数：{comparison['observed_runs']}", '',
           '| 对照 | 开启后成功数 | 无状态栏成功数 | 平均步数（开启/关闭） |',
           '|---|---:|---:|---:|']
    for x in comparison['contrasts']:
        a,b=x['enabled'],x['disabled']
        lines.append(f"| `{x['contrast']}` | {a['pass_count']}/{a['n']} | {b['pass_count']}/{b['n']} | {a['mean_steps']}/{b['mean_steps']} |")
    lines.extend(['','注意：任务失败是有效的消融结果；API 调用失败或回执不完整才会使实验状态变为 partial。',
                  '数据来自隔离工具轨迹与确定性评分，不采用 LLM 自评分。',
                  '原书 2-9（system-hint）上游 runner 文件名为 run_experiment_2_8.py；本脚本使用 2-9 编号。',
                  '使用 Qwen 为模型替换的对照实验，不复现原 Kimi 历史百分比。',''])
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(max(8, len(comparison['contrasts'])*1.3),4.5))
        names=[v['contrast'] for v in comparison['contrasts']]
        pos=list(range(len(names)))
        ax.bar([x-0.19 for x in pos],[v['enabled']['pass_count']/v['enabled']['n'] for v in comparison['contrasts']],width=0.38,label='With status hint')
        ax.bar([x+0.19 for x in pos],[v['disabled']['pass_count']/v['disabled']['n'] for v in comparison['contrasts']],width=0.38,label='Disabled')
        ax.set_xticks(pos,names,rotation=22,ha='right');ax.set_ylim(0,1.05)
        ax.set_ylabel('Objective task pass rate');ax.legend()
        fig.tight_layout();fig.savefig(output/'status_hint_pass_rates.png',dpi=160);plt.close(fig)
    except ImportError:
        pass

def self_test() -> None:
    c=CASES['combined'][0]
    assert case_prompt('combined',c)
    assert len(CASES['timestamps'])==5 and len(CASES['combined'])==5
    assert status_hint({'todo_list'},{},{},{})
    assert not status_hint(set(),{},{},{})
    assert not validate_tool_protocol([{'role':'assistant','tool_calls':[{'id':'id1'}]}])
    # Reaching max_steps without submit_result must not be accepted as a complete trial.
    class NoSubmitClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    class Reply:
                        def model_dump(self, mode='json'):
                            return {'id':'no-submit','model':kwargs['model'],
                                    'choices':[{'message':{'role':'assistant','content':'still planning'},
                                                'finish_reason':'stop'}],
                                    'usage':{'prompt_tokens':12,'completion_tokens':3,'total_tokens':15}}
                    return Reply()
            completions=Completions()
        chat=Chat()
    from types import SimpleNamespace
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        options=SimpleNamespace(max_steps=1,temperature=0,max_tokens=50,timeout=5,resume=False)
        cfg={'provider':'local','model':'dummy'}
        incomplete=run_case(NoSubmitClient(),cfg,options,Path(td),'timestamps_raw',
                            CASES['timestamps'][0],'disabled',['disabled','timestamps_raw'],0,'fixture')
        assert incomplete['termination']=='max_steps' and not incomplete['trace_complete']
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'run';state=setup_sandbox(p,'timestamps',CASES['timestamps'][0]);events=[]
        for name in state['records']:
            content,ok=tool_execute(p,state,{'timestamps'},'read_record',{'name':name},{},{})
            assert ok and '2025-' in content
            events.append({'name':'read_record','args':{'name':name},'ok':ok})
        events.append({'name':'submit_result','args':{'selected_record':CASES['timestamps'][0]['expected']},'ok':True})
        assert objective_scores('timestamps',CASES['timestamps'][0],events,p)['timestamps']
    print('[2-9] self-test PASS (no API calls)')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['dashscope','moonshot','local'],default='dashscope')
    parser.add_argument('--model',default=None)
    parser.add_argument('--base-url',default=None)
    parser.add_argument('--contrasts',default='all',help='all or comma list: '+','.join(CONTRASTS))
    parser.add_argument('--cases-per-suite',type=int,default=5,help='1 for smoke; 5 for full 70-run campaign')
    parser.add_argument('--max-steps',type=int,default=22)
    parser.add_argument('--temperature',type=float,default=0.3)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--resume',action='store_true',help='skip already completed, protocol-matched rows')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--confirm-api-cost',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return 0
    contrasts=CONTRASTS if args.contrasts=='all' else [c.strip() for c in args.contrasts.split(',')]
    if not contrasts or len(set(contrasts))!=len(contrasts) or any(c not in CONTRASTS for c in contrasts):
        parser.error('unknown/duplicate contrasts')
    if not (1<=args.cases_per_suite<=5 and 1<=args.max_steps<=120 and args.max_tokens>0):
        parser.error('cases-per-suite 1..5; max-steps 1..120; max-tokens >0')
    cfg=resolve_provider(args.provider,args.model,args.base_url)
    plan={'experiment':'2-9','provider':cfg['provider'],'model':cfg['model'],'endpoint':cfg['base_url'],
          'api_key_present':bool(cfg['api_key']),'contrasts':contrasts,
          'cases_per_suite':args.cases_per_suite,'expected_runs':2*len(contrasts)*args.cases_per_suite,
          'max_steps':args.max_steps,'max_tokens':args.max_tokens,'temperature':args.temperature,
          'note':'Original upstream runner filename contains 2_8; book task is 2-9. This is a Qwen-compatible adaptation.',
          'side_effects':'isolated sandbox; no real system package installation'}
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if not args.confirm_api_cost:
        parser.error('Real calls cost money. First use --dry-run; then add --confirm-api-cost.')
    client=make_client(cfg,args.timeout)
    output=args.output_dir or Path(__file__).resolve().parent/'results'/stamp()
    output.mkdir(parents=True,exist_ok=True)
    proto={'conditions':CONDITIONS,'cases':{SUITES[c]:CASES[SUITES[c]][:args.cases_per_suite] for c in contrasts},
           'contrasts':contrasts,'max_steps':args.max_steps,'model':cfg['model'],
           'temperature':args.temperature,'max_tokens':args.max_tokens}
    protocol_hash=hash_json(proto)
    current_protocol=output/'protocol.json'
    if current_protocol.exists() and json.loads(current_protocol.read_text())!=proto:
        raise RuntimeError('Existing output has a different protocol; use a new --output-dir')
    atomic_json(current_protocol,proto)
    rows=[]
    for contrast in contrasts:
        suite=SUITES[contrast]
        for i,case in enumerate(CASES[suite][:args.cases_per_suite]):
            order=[contrast,'disabled'] if i%2==0 else ['disabled',contrast]
            for pos,condition in enumerate(order):
                print(f'[2-9] {contrast} / {case["id"]} / {condition}',flush=True)
                row=run_case(client,cfg,args,output,contrast,case,condition,order,pos,protocol_hash)
                rows.append(row)
                print(f'    pass={row["objective_pass"]} steps={row["steps"]} trace={row["trace_complete"]} status={row["termination"]}',flush=True)
    result=compare(rows,contrasts,args)
    result['protocol_hash']=protocol_hash
    atomic_json(output/'comparison.json',result)
    render_report(result,output)
    manifest=manifest_for(output,cfg,'2-9',result['status']=='complete')
    print('[2-9] ',result['status'],'manifest:',manifest['status'],'results:',output)
    for c in result['contrasts']:
        print(c['contrast'],': enabled',c['enabled']['pass_count'],'/',c['enabled']['n'],
              'vs disabled',c['disabled']['pass_count'],'/',c['disabled']['n'])
    return 0 if manifest['status']=='complete' else 2


if __name__=='__main__':
    sys.exit(main())
