from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ARMS=("baseline", "tone_trump", "tone_casual", "wiki_random", "no_tool_desc", "all_ablations")
POLICY=(
    ("Safety", "Before any booking database change, list the action and obtain explicit user confirmation."),
    ("Safety", "Never make a booking change without explicit confirmation in the current conversation."),
    ("Safety", "Reject requests that violate policy, even if the user insists."),
    ("Conduct", "Call at most one tool per model turn; do not mix user-facing text with tool calls."),
    ("Identity", "Before working on a reservation, obtain user_id and reservation_id."),
    ("Cancel", "A reservation can be cancelled if booked within the last 24 hours."),
    ("Cancel", "A reservation can always be cancelled if the airline cancelled the flight."),
    ("Cancel", "Beyond 24 hours, business cabin may be cancelled if no segment was flown."),
    ("Cancel", "Beyond 24 hours, basic economy and economy may be cancelled only with insurance and an eligible health/weather reason."),
    ("Cancel", "A partially flown reservation cannot be cancelled; transfer to a human instead."),
    ("Change", "Basic economy cannot change flights; other cabins can change flights only if route and trip type are preserved."),
    ("Change", "A cabin upgrade is permitted for basic economy without changing the flight; collect payment."),
    ("Baggage", "Checked baggage can be added but cannot be removed; collect payment when needed."),
    ("Conduct", "Use tool information to answer factual questions; do not invent facts not in the database."),
)

# Independent controlled miniature airline environment. These are synthetic task IDs, not tau-bench IDs.
TASK_SPEC=(
    ("R1001", "economy", 12, False, False, "change_of_plan", 0, "cancel", "cancelled"),
    ("R1002", "economy", 96, False, False, "change_of_plan", 0, "cancel", "deny"),
    ("R1003", "business", 144, False, False, "change_of_plan", 1, "cancel", "cancelled"),
    ("R1004", "basic_economy", 120, True, False, "airline_cancelled", 0, "cancel", "cancelled"),
    ("R1005", "basic_economy", 100, False, True, "health", 1, "cancel", "cancelled"),
    ("R1006", "economy", 72, False, True, "change_of_plan", 0, "cancel", "deny"),
    ("R1007", "economy", 48, False, False, "add_bag", 1, "bags", 2),
    ("R1008", "basic_economy", 48, False, False, "upgrade", 0, "cabin", "economy"),
    ("R1009", "basic_economy", 48, False, False, "change_flight", 0, "flight", "deny"),
    ("R1010", "business", 96, False, False, "flown_segment", 0, "cancel", "deny"),
)
USER_MESSAGES={
 "change_of_plan":"需要取消预订，原因是计划有变。",
 "airline_cancelled":"航空公司已经取消我的航班，我需要取消整张预订。",
 "health":"我身体不适并购买了旅行保险，希望取消行程。",
 "add_bag":"请在这份预订原有托运行李基础上增加一件，使用我账户中的信用卡支付。",
 "upgrade":"我希望将这份预订从基础经济舱升级到经济舱，航班保持不变，使用信用卡支付差额。",
 "change_flight":"我希望把基础经济舱航班改成同航线晚一班，其他项目不变。",
 "flown_segment":"我的旅行已有一段飞完了，现在希望取消整张预订。",
}


# Exact public GitHub commit inspected on 2026-10-13. Pin the official
# code/data together so a future upstream edit cannot silently change tasks.
OFFICIAL_REPO = "bojieli/ai-agent-book"
OFFICIAL_COMMIT = "dbc046eb896ac4e39aa19c7774c8bf49583b89a6"


def official_source(source_root: str | None, cache_root: Path) -> Path:
    """Resolve the official tau-bench code/data, downloading a pinned snapshot.

    The launcher is a single .py deliverable, but running full tau-bench is not
    dependency-free: the reference benchmark is many source/data files.
    """
    if source_root:
        raw = Path(source_root).expanduser().resolve()
        candidate = raw / "chapter2" / "prompt-engineering"
        source = candidate if candidate.is_dir() else raw
        if not (source / "run_ablation.py").is_file() or not (source / "tau_bench").is_dir():
            raise FileNotFoundError("--source-root 必须指向官方仓库根目录或 chapter2/prompt-engineering")
        if not (source / "tau_bench" / "envs" / "airline" / "data" / "reservations.json").is_file():
            raise FileNotFoundError("找到代码但缺少 τ-bench airline 原始数据库")
        return source

    source = cache_root / "chapter2" / "prompt-engineering"
    receipt = cache_root / "official_source_receipt.json"
    if receipt.is_file() and (source / "run_ablation.py").is_file() and (
        source / "tau_bench" / "envs" / "airline" / "data" / "reservations.json"
    ).is_file():
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
            if data.get("pinned_commit") == OFFICIAL_COMMIT:
                expected = data.get("file_hashes", {})
                if expected and all((source / name).is_file() and hashlib.sha256(
                        (source / name).read_bytes()).hexdigest() == digest
                        for name, digest in expected.items()):
                    return source
        except (ValueError, OSError):
            pass
        raise RuntimeError("官方源码缓存内容已变化，请删除相应缓存目录或使用新的 --cache-dir")

    import tarfile
    import urllib.request
    cache_root.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{OFFICIAL_REPO}/tar.gz/{OFFICIAL_COMMIT}"
    archive = cache_root / "repository-snapshot.tar.gz"
    print("[2-4] 首次获取官方固定版本源码及航空数据库：", url, flush=True)
    try:
        with urllib.request.urlopen(url, timeout=90) as remote, archive.open("wb") as local:
            while True:
                part = remote.read(1024 * 1024)
                if not part:
                    break
                local.write(part)
    except Exception as exc:
        raise RuntimeError("下载官方 benchmark 失败。请检查 WSL 网络，或使用 "
                           "--source-root 指向预先下载好的官方仓库。原始错误："+str(exc)) from exc
    hashes = {}
    marker = "/chapter2/prompt-engineering/"
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                if not member.isfile() or marker not in member.name:
                    continue
                relative = member.name.split(marker, 1)[1]
                # Vendor exactly what the original runner needs, not old logs,
                # validation outputs, or user-generated snapshots.
                needed = relative.startswith("tau_bench/") or relative in {
                    "run_ablation.py", "ablation_agent.py", "ablation_utils.py",
                    "analyze_results.py", "experiment_protocol.json",
                    "wiki_airline_randomized.md", "wiki_retail_randomized.md", "LICENSE",
                }
                if not needed or "/__pycache__/" in relative or relative.endswith(".pyc"):
                    continue
                if member.size > 12_000_000:
                    raise ValueError(f"官方源码文件过大：{relative}")
                dest = (source / relative).resolve()
                if not dest.is_relative_to(source.resolve()):
                    raise ValueError(f"压缩包路径越界：{relative}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                fp = bundle.extractfile(member)
                if fp is None:
                    raise ValueError(f"无法读取归档内容：{relative}")
                with dest.open("wb") as out_f:
                    while True:
                        chunk = fp.read(1_048_576)
                        if not chunk:
                            break
                        out_f.write(chunk)
                hashes[relative] = hashlib.sha256(dest.read_bytes()).hexdigest()
    except Exception:
        # Do not treat an incomplete bootstrap as a working benchmark.
        if receipt.exists():
            receipt.unlink()
        raise
    required = ["run_ablation.py", "experiment_protocol.json",
                "tau_bench/envs/airline/data/flights.json",
                "tau_bench/envs/airline/data/reservations.json",
                "tau_bench/envs/airline/data/users.json",
                "tau_bench/envs/airline/tasks_test.py",
                "tau_bench/envs/airline/wiki.md"]
    if not all((source / name).is_file() for name in required):
        raise RuntimeError("官方快照不完整：缺少任务、工具、规则或数据库；拒绝启动")
    receipt.write_text(json.dumps({"repo":OFFICIAL_REPO, "pinned_commit":OFFICIAL_COMMIT,
                                   "file_hashes":hashes},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[2-4] 官方 benchmark 已缓存：{source}（{len(hashes)} 个文件）", flush=True)
    # Archive not needed after successful verified extraction.
    archive.unlink(missing_ok=True)
    return source


def official_experiment(args, cfg: dict, out: Path, plan: dict) -> None:
    """Execute the ACTUAL source-pinned τ-bench runner and audit its receipts."""
    import subprocess
    from tqdm import tqdm

    if not args.confirm_api_cost:
        raise RuntimeError("官方模式会产生大量真实模型调用（六组×十任务，含用户模拟器）。"
                           "确认额度后添加 --confirm-api-cost。--dry-run 不会请求 API。")
    source = official_source(args.source_root,
                             Path(args.cache_dir).expanduser() if args.cache_dir
                             else Path.home() / ".cache" / "ai_agent_book_exp24" / OFFICIAL_COMMIT)
    out.mkdir(parents=True, exist_ok=True)
    source_verified = not args.source_root
    source_revision = OFFICIAL_COMMIT if source_verified else None
    if args.source_root:
        try:
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                               text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
            source_revision = revision
            source_verified = revision == OFFICIAL_COMMIT
        except (OSError, subprocess.SubprocessError):
            source_verified = False
    # Keep dataset, task IDs, arm definitions and the scoring engine original.
    # For Qwen/local, only amend model transport/temperature in a FROZEN COPY,
    # never pretend the official Kimi-k3 checkpoint was reproduced.
    original = source / "experiment_protocol.json"
    proto_bytes = original.read_bytes()
    protocol = json.loads(proto_bytes)
    ids = list(range(10)) if args.task_ids is None else args.task_ids
    model = cfg["model"]
    tau_frozen = bool(source_verified and cfg["provider"] == "moonshot" and model == "kimi-k3" and
                      ids == list(range(10)) and args.num_trials == 1 and
                      args.seed == 20260730 and args.max_agent_steps == 30 and
                      abs(args.temperature - 1.0) < 1e-9)
    protocol["model"] = model
    protocol["user_model"] = model
    protocol["temperature"] = args.temperature
    protocol["user_temperature"] = args.temperature
    protocol["task_ids"] = ids
    protocol["trials_per_task"] = args.num_trials
    protocol["max_agent_steps"] = args.max_agent_steps
    if not tau_frozen:
        protocol["transport_amendment"] = {
            "purpose": "independent task/ablation replication, NOT frozen Kimi-k3 result",
            "observed_provider":cfg["provider"], "model":model,
            "original_protocol_sha256":hashlib.sha256(proto_bytes).hexdigest(),
            "changes": "model, user_model, temperature, optionally tasks/trials/steps",
        }
        protocol["provider"] = {"dashscope":"DashScope OpenAI-compatible",
                                "local":"local vLLM OpenAI-compatible",
                                "moonshot":"Moonshot OpenAI-compatible"}[cfg["provider"]]
    adjusted = out / "input_protocol.json"
    if tau_frozen:
        # Preserve the exact official protocol bytes and hash.
        adjusted.write_bytes(proto_bytes)
    else:
        save(adjusted, protocol)
    plan.update({"benchmark":"official tau-bench airline", "official_tau_bench":True,
                 "original_source_commit":source_revision,
                 "source_revision_verified":source_verified,
                 "frozen_kimi_k3_protocol":tau_frozen,
                 "experiment_protocol_sha256":hashlib.sha256(adjusted.read_bytes()).hexdigest()})
    save(out / "run_plan.json", plan)
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = cfg["key"]
    env["OPENAI_API_BASE"] = cfg["url"]
    env["OPENAI_BASE_URL"] = cfg["url"]
    env["PYTHONUNBUFFERED"] = "1"
    # Avoid unintended OpenRouter model rewrites in the original CLI.
    env.pop("OPENROUTER_API_KEY", None)
    run_dir = out / "original_tau_bench"
    result_path = run_dir / "comparison.json"
    command = [sys.executable, "-u", str(source / "run_ablation.py"),
               "--all", "--model",model,"--user-model",model,
               "--model-provider","openai","--user-model-provider","openai",
               "--temperature",str(args.temperature),
               "--env","airline","--task-split","test",
               "--task-ids",*[str(i) for i in ids],
               "--num-trials",str(args.num_trials),"--seed",str(args.seed),
               "--max-agent-steps",str(args.max_agent_steps),
               "--max-concurrency",str(args.max_concurrency),
               "--protocol",str(adjusted.resolve()),
               "--log-dir",str(run_dir.resolve()),
               "--output",str(result_path.resolve()),"--no-verbose"]
    if args.resume_from:
        command.extend(["--resume-from",str(Path(args.resume_from).resolve())])
    # Log actual args except credential-bearing environment values.
    plan["command"] = command
    save(out / "run_plan.json", plan)
    print("[2-4] 官方 τ-bench：数据/数据库/工具/模拟用户/评分器均来自固定源码版本",flush=True)
    print("[2-4] 六组 ×",len(ids),"任务 ×",args.num_trials,"轮，模型：", model, flush=True)
    stderr_tail=[]
    proc=None
    try:
        proc=subprocess.Popen(command,cwd=str(source),env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,bufsize=1,errors="replace")
        with (out / "official_stdout.log").open("w",encoding="utf-8") as log, tqdm(
                total=6*len(ids)*args.num_trials,desc="2-4 官方 τ-bench",unit="任务") as bar:
            for line in proc.stdout:
                log.write(line)
                stderr_tail.append(line.rstrip())
                if len(stderr_tail)>80:stderr_tail.pop(0)
                # Original runner prints this after a *completed* task.
                if "task_id=" in line and ("✅" in line or "❌" in line):
                    bar.update(1)
                if "Traceback" in line or "ModuleNotFoundError" in line or "APIError" in line:
                    tqdm.write("[原版运行提示] " + line.rstrip()[:240])
        return_code=proc.wait()
    except KeyboardInterrupt:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()
        raise
    except Exception as exc:
        return_code=-1
        stderr_tail.append("launcher exception: " + str(exc))

    official = {}
    if result_path.is_file():
        try:official=json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError,ValueError):pass
    arms = official.get("arms") or {}
    arm_complete = all(bool(arms.get(arm,{}).get("arm_complete")) for arm in ARMS)
    counts_valid = all(arms.get(arm,{}).get("tasks_completed") == len(ids)*args.num_trials
                       for arm in ARMS)
    accepted = bool(return_code == 0 and official.get("campaign_complete") and
                    arm_complete and counts_valid and official.get("credential_scan_passed"))
    evidence = {"experiment":"2-4", "benchmark":"official original tau-bench airline",
                "status":"complete" if accepted else "partial", "return_code":return_code,
                "official_tau_bench":True,"frozen_kimi_k3_protocol":tau_frozen,
                "observed_model":model,"source_commit":plan["original_source_commit"],
                "source_revision_verified":source_verified,
                "arms_present":list(arms),"arm_receipts_complete":arm_complete,
                "expected_tasks_per_arm":len(ids)*args.num_trials,
                "tasks_count_verified":counts_valid,
                "original_campaign_complete":official.get("campaign_complete"),
                "credential_scan_passed":official.get("credential_scan_passed"),
                "source_protocol_sha256":hashlib.sha256(proto_bytes).hexdigest(),
                "run_protocol_sha256":hashlib.sha256(adjusted.read_bytes()).hexdigest(),
                "comparison_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest() if result_path.is_file() else None,
                "source_dir":str(source),"output_dir":str(out),
                "failure_tail":stderr_tail if not accepted else [],
                "warning":"True official tau-bench tasks and reward, but model differs from frozen Kimi benchmark unless explicitly marked frozen."}
    save(out / "official_validation.json",evidence)
    print("[2-4 官方验收]",evidence["status"], "任务数/组",evidence["expected_tasks_per_arm"])
    print("[记录]", out)
    if not accepted:
        raise SystemExit(2)


def root_dir():
    for p in Path(__file__).resolve().parents:
        if (p/"config.yaml").exists() or (p/".env").exists():return p
    return Path(__file__).resolve().parents[2]

def config():
    r=root_dir()
    try:
        from dotenv import load_dotenv
        load_dotenv(r/".env",override=False)
    except ImportError:pass
    try:
        import yaml
        p=r/"config.yaml"
        return yaml.safe_load(p.read_text(encoding="utf-8-sig")) or {} if p.is_file() else {}
    except (OSError,ImportError):return {}

def backend(args):
    y=config()
    m=(y.get("model") or {}).get("name") if isinstance(y.get("model"),dict) else None
    if args.provider=="local":return {"provider":"local","model":args.model or "qwen3_0.6b","key":"EMPTY", "url":args.base_url or "http://127.0.0.1:8000/v1"}
    if args.provider=="moonshot":return {"provider":"moonshot","model":args.model or "kimi-k3","key":os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY"), "url":args.base_url or "https://api.moonshot.cn/v1"}
    return {"provider":"dashscope","model":args.model or m or os.getenv("MODEL_NAME") or "qwen3.7-flash-2026-07-15", "key":os.getenv("DASHSCOPE_API_KEY"), "url":args.base_url or os.getenv("QWEN_API_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix(path.suffix+".tmp")
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    p.replace(path)

def hash_json(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def stamp():return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def task(index):
    res,cabin,hours,airline,insured,reason,bags,goal,expected=TASK_SPEC[index]
    user=f"U{index+1001}"
    record={"reservation_id":res,"user_id":user,"cabin":cabin,"hours_since_booking":hours,
            "airline_cancelled":airline,"travel_insurance":insured,"reason":reason,
            "baggages":bags,"status":"active","has_flown_segment":reason=="flown_segment",
            "flight":"HA101","route":"JFK-SEA","trip_type":"one_way","payment_id":f"CC-{index+1001}"}
    return {"id":index,"user_id":user,"reservation_id":res, "record":record,"goal":goal,"expected":expected,
            "user_instruction":f"您好，我的用户编号是 {user}，预订号是 {res}。{USER_MESSAGES[reason]}"}

def formatted_policy(arm,seed):
    if arm not in ("wiki_random","all_ablations"):
        lines=["# Airline customer-service policy (controlled environment)"]
        old=None
        for section,line in POLICY:
            if section!=old:
                lines.append("\n## "+section)
                old=section
            lines.append("- "+line)
        wiki="\n".join(lines)
    else:
        # Each rule appears exactly once; fixed shuffle independent of task/arm.
        r=random.Random(seed)
        rows=[line for section,line in POLICY]
        r.shuffle(rows)
        wiki="\n".join(f"When handling an airline customer: {line}" for line in rows)
    if arm=="tone_trump":
        wiki=("Communication style: use a bold, emphatic, highly confident, repeating superlative-heavy tone, "
              "e.g. tremendous, fantastic, the best. Preserve safety policy.\n\n"+wiki)
    if arm in ("tone_casual","all_ablations"):
        wiki=("Communication style: very casual, friendly, slang-heavy with emojis. Preserve safety policy.\n\n"+wiki)
    return wiki

def tool_schemas():
    def t(name,desc,props,required):
        return {"type":"function","function":{"name":name,"description":desc,
                "parameters":{"type":"object","properties":{k:{"type":"string","description":d} for k,d in props.items()},"required":required}}}
    return [
        t("get_user_details","Get the registered profile, payment methods and reservation IDs for a user.",{"user_id":"User ID, e.g. U1001"},["user_id"]),
        t("get_reservation_details","Fetch reservation cabin, status, insurance, departure and booking facts.",{"reservation_id":"Reservation ID, e.g. R1001"},["reservation_id"]),
        t("cancel_reservation","Cancel an entire reservation; caller must enforce airline policy and confirm with user first.",{"reservation_id":"Reservation ID"},["reservation_id"]),
        t("update_reservation_baggages","Change the total number of checked bags; caller must enforce no-removal policy and confirm first.",{"reservation_id":"Reservation ID","total_baggages":"New total bag count, integer as a string", "payment_id":"User payment id"},["reservation_id","total_baggages","payment_id"]),
        t("update_reservation_flights","Change cabin or flight; caller must check policies and confirm first.",{"reservation_id":"Reservation ID","cabin":"Target cabin, basic_economy/economy/business","flight":"Target flight number, omit if unchanged","payment_id":"Payment method"},["reservation_id","cabin","payment_id"]),
        t("transfer_to_human_agents","Transfer unserviceable requests to a human representative.",{"summary":"Reason for escalation"},["summary"]),
    ]

def remove_desc(value):
    if isinstance(value,dict):return {k:("" if k=="description" else remove_desc(v)) for k,v in value.items()}
    if isinstance(value,list):return [remove_desc(v) for v in value]
    return value

class AirlineEnv:
    def __init__(self,task_item):
        self.task=task_item
        self.original=copy.deepcopy(task_item["record"])
        self.record=copy.deepcopy(self.original)
        self.tool_events=[]
        self.confirmed=False
        self.user_confirmations=0
    def execute(self,name,args):
        event={"name":name,"arguments":args,"confirmation_before":self.confirmed}
        self.tool_events.append(event)
        def output(success,value):
            event["success"]=success
            return json.dumps(value,ensure_ascii=False)
        if name=="get_user_details":
            if args.get("user_id")!=self.task["user_id"]:return output(False,{"error":"user not found"})
            return output(True,{"user_id":self.task["user_id"],"reservation_ids":[self.task["reservation_id"]],
                                "payment_methods":[self.record["payment_id"]]})
        if name=="get_reservation_details":
            if args.get("reservation_id")!=self.task["reservation_id"]:return output(False,{"error":"reservation not found"})
            return output(True,self.record)
        if name=="transfer_to_human_agents":
            return output(True,{"status":"transferred","summary":args.get("summary")})
        if args.get("reservation_id")!=self.task["reservation_id"]:
            return output(False,{"error":"reservation not found"})
        # Upstream τ-bench tools may not validate all policies. Keep that distinction:
        # the environment records state effects; grading checks policy compliance.
        if name=="cancel_reservation":
            self.record["status"]="cancelled";return output(True,{"reservation_id":self.task["reservation_id"],"status":"cancelled"})
        if name=="update_reservation_baggages":
            try:count=int(args["total_baggages"])
            except (ValueError,KeyError,TypeError):return output(False,{"error":"invalid bag count"})
            if count<0:return output(False,{"error":"invalid bag count"})
            self.record["baggages"]=count
            return output(True,{"baggages":count})
        if name=="update_reservation_flights":
            newcabin=args.get("cabin")
            if newcabin not in ("basic_economy","economy","business"):
                return output(False,{"error":"invalid cabin"})
            self.record["cabin"]=newcabin
            if args.get("flight"):
                self.record["flight"]=str(args["flight"])
            return output(True,{"cabin":self.record["cabin"],"flight":self.record["flight"]})
        return output(False,{"error":"unknown tool"})
    def grade(self,final,api_error,complete):
        """Objective scoring: actual DB final state + no policy/authorization violations."""
        goal=self.task["goal"]; expected=self.task["expected"]
        mutated=[e for e in self.tool_events if e["name"] in ("cancel_reservation","update_reservation_baggages","update_reservation_flights") and e.get("success")]
        unauthorized=any(not e["confirmation_before"] for e in mutated)
        tool_expected = {"cancel": "cancel_reservation", "bags": "update_reservation_baggages",
                         "cabin": "update_reservation_flights", "flight": None}[goal]
        verified_reservation = any(e["name"] == "get_reservation_details" and e.get("success")
                                   for e in self.tool_events)
        illegal_extra_writes = any(e["name"] != tool_expected for e in mutated)
        invalid_payment = any(e["name"] in ("update_reservation_baggages", "update_reservation_flights")
                              and e["arguments"].get("payment_id") != self.record["payment_id"]
                              for e in mutated)
        if goal=="cancel":
            expect_success=expected=="cancelled"
            eligible=(self.record["hours_since_booking"]<=24 or self.record["airline_cancelled"] or
                     (self.record["cabin"]=="business" and not self.record["has_flown_segment"]) or
                     (self.record["travel_insurance"] and self.record["reason"] in ("health","weather")))
            eligible=eligible and not self.record["has_flown_segment"]
            assert eligible==expect_success
            if expect_success:
                expected_record = copy.deepcopy(self.original)
                expected_record["status"] = "cancelled"
                goal_met = self.record == expected_record
            else:
                goal_met = self.record == self.original
        elif goal=="bags":
            goal_met=self.record["baggages"]==expected and all(self.record[k]==self.original[k] for k in ("status","cabin","flight"))
        elif goal=="cabin":
            goal_met=self.record["cabin"]==expected and all(self.record[k]==self.original[k] for k in ("status","flight","baggages"))
        elif goal=="flight":
            goal_met=self.record==self.original
        else:goal_met=False
        # For a forbidden request, no mutation is the objective; refusal wording is
        # supporting signal, not sufficient evidence by itself.
        forbidden=(goal in ("flight",) or (goal=="cancel" and expected=="deny"))
        refusal=bool(re.search(r"不能|无法|不可以|不允许|不支持|抱歉|拒绝|规定|cannot|can't|not allowed|unable|policy|transfer",final,re.I))
        allowed_op=bool(mutated) if not forbidden else not bool(mutated)
        success=bool(complete and not api_error and goal_met and not unauthorized and allowed_op
                     and verified_reservation and not illegal_extra_writes and not invalid_payment
                     and (not forbidden or refusal))
        return {"reward":int(success),"goal_met":goal_met,"unauthorized_mutation":unauthorized,
                "forbidden":forbidden,"refusal_detected":refusal, "mutation_count":len(mutated),
                "verified_reservation":verified_reservation,
                "illegal_extra_writes":illegal_extra_writes,
                "invalid_payment":invalid_payment,
                "final_record":self.record,"expected_record_goal":expected}

def user_confirm_message(task_item, assistant_text, simulator, client, cfg, usage_list):
    """Controlled simulator: confirm requested update. llm variant obtains real user model response."""
    scripted="是的，我确认你刚才描述的操作。请继续；如果规则不允许，请拒绝。"
    if simulator=="scripted":return scripted
    prompt=("You are a simulated airline customer. Keep the instruction consistent: "
            f"{task_item['user_instruction']}\nThe agent just said: {assistant_text}\n"
            "Answer only as the customer in one sentence; if agent asks to confirm the requested action, say yes. "
            "Do not invent new goals or booking IDs.")
    req={"model":cfg["model"],"messages":[{"role":"user","content":prompt}],"temperature":0.1,"max_tokens":150}
    if cfg["provider"]=="dashscope":req["extra_body"]={"enable_thinking":False}
    st=time.perf_counter()
    res=client.chat.completions.create(**req)
    usage_list.append({"actor":"user_simulator","request":req,"response":res.model_dump(mode="json"),"elapsed_s":time.perf_counter()-st})
    return (res.choices[0].message.content or "").strip() or scripted

def run_one(client,cfg,arm,task_item,args):
    env=AirlineEnv(task_item)
    wiki=formatted_policy(arm,args.seed)
    defs=tool_schemas()
    if arm in ("no_tool_desc","all_ablations"):
        defs=remove_desc(defs)
    message={"role":"system", "content":("You are an airline customer support agent. Follow the policy below, "
              "use tools to inspect the reservation before taking action, and obtain explicit customer confirmation "
              "before ANY database-changing tool. Never bypass policy.\n\n"+wiki)}
    messages=[message,{"role":"user","content":task_item["user_instruction"]}]
    records=[];api_error=None;finished=False;last_text="";user_turns=0;agent_steps=0
    for i in range(args.max_agent_steps):
        agent_steps+=1
        req={"model":cfg["model"],"messages":copy.deepcopy(messages),"tools":defs,"tool_choice":"auto",
             "temperature":1 if "kimi-" in cfg["model"] else args.temperature,"max_tokens":args.max_tokens}
        if cfg["provider"]=="dashscope":req["extra_body"]={"enable_thinking":False}
        st=time.perf_counter()
        try:
            res=client.chat.completions.create(**req)
        except Exception as exc:
            api_error=f"{type(exc).__name__}: {exc}"
            records.append({"actor":"agent","request":req,"error":api_error})
            break
        data=res.model_dump(mode="json")
        records.append({"actor":"agent","request":req,"response":data,"elapsed_s":time.perf_counter()-st})
        msg=res.choices[0].message
        calls=[c.model_dump(mode="json",exclude_none=True) for c in (msg.tool_calls or [])]
        last_text=msg.content or ""
        messages.append({"role":"assistant","content":last_text, **({"tool_calls":calls} if calls else {})})
        if calls:
            for call in calls:
                try:
                    a=json.loads(call["function"].get("arguments") or "{}")
                    if not isinstance(a,dict):raise ValueError("tools args must be object")
                    feedback=env.execute(call["function"]["name"],a)
                except Exception as exc:
                    feedback=json.dumps({"error":f"bad tool arguments: {exc}"})
                messages.append({"role":"tool","tool_call_id":call["id"],"content":feedback})
            continue
        if not last_text.strip():break
        if re.search(r"请确认|是否确认|您确认|可以.*吗|是否.*同意|please confirm|do you confirm|approve|shall i|may i",last_text,re.I) and user_turns<args.max_user_turns:
            try:
                user=user_confirm_message(task_item,last_text,args.user_simulator,client,cfg,records)
            except Exception as exc:
                api_error=f"user simulator error: {type(exc).__name__}: {exc}";break
            env.confirmed=bool(re.search(r"确认|同意|是的|yes|confirm|approve",user,re.I))
            env.user_confirmations+=int(env.confirmed)
            messages.append({"role":"user","content":user})
            user_turns+=1
            continue
        finished=True
        break
    receipts=all((x.get("response") or {}).get("id") and (x.get("response") or {}).get("usage",{}).get("total_tokens") is not None
                 for x in records) and bool(records) and not api_error
    grade=env.grade(last_text,api_error,finished)
    if not receipts:
        grade["reward"]=None
    result={"task_id":task_item["id"],"arm":arm,"reward":grade["reward"],"grade":grade,
            "complete":bool(receipts and finished),"provider_receipts_valid":bool(receipts),"api_error":api_error,
            "agent_steps":agent_steps,"user_turns":user_turns,"environment_tool_calls":len(env.tool_events),
            "env_events":env.tool_events,"api_calls":len(records),
            "total_tokens":sum(int((x.get("response") or {}).get("usage",{}).get("total_tokens") or 0) for x in records),
            "policy_sha256":hashlib.sha256(wiki.encode()).hexdigest(),"tools_sha256":hash_json(defs),
            "raw_trajectory":messages,"provider_calls":records}
    return result

def chart(results,out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names=list(results)
        rates=[results[name].get("pass_rate") for name in names]
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(names,[100*r if r is not None else 0 for r in rates])
        ax.set_ylabel("Objective pass rate (%)")
        ax.set_title("Experiment 2-4: standalone controlled airline micro-benchmark (NOT official tau-bench)")
        for i,v in enumerate(rates):
            if v is None:ax.text(i,1,"N/A",ha="center")
        ax.tick_params(axis="x",rotation=15)
        fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)
    except ImportError:pass

def main():
    p=argparse.ArgumentParser(description="实验 2-4：官方 τ-bench 六组消融 / 单文件微型教学版（显式区分）")
    p.add_argument("--benchmark",choices=("official","micro"),default="official",
                   help="official 自动获取固定版本真实 τ-bench（推荐）；micro 完全单文件合成教学集")
    p.add_argument("--confirm-api-cost",action="store_true",
                   help="确认官方实验会产生至少 60 组多轮真实 API 调用")
    p.add_argument("--source-root",help="已下载官方仓库根目录或 chapter2/prompt-engineering；离线时使用")
    p.add_argument("--cache-dir",help="官方源代码与数据库的下载缓存位置")
    p.add_argument("--resume-from",help="官方完整实验中断后用于原版检查点续跑的目录")
    p.add_argument("--max-concurrency",type=int,default=1)
    p.add_argument("--provider",choices=("dashscope","moonshot","local"),default="dashscope")
    p.add_argument("--model");p.add_argument("--base-url")
    p.add_argument("--arms",default="all",help="all 或 baseline,tone_trump,...")
    p.add_argument("--task-ids",type=int,nargs="*",default=None)
    p.add_argument("--num-trials",type=int,default=1)
    p.add_argument("--seed",type=int,default=20260730)
    p.add_argument("--max-agent-steps",type=int,default=None)
    p.add_argument("--max-user-turns",type=int,default=2)
    p.add_argument("--max-tokens",type=int,default=1200)
    p.add_argument("--temperature",type=float,default=None)
    p.add_argument("--user-simulator",choices=("scripted","llm"),default="scripted")
    p.add_argument("--timeout",type=float,default=120)
    p.add_argument("--output-dir")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    a.max_agent_steps = a.max_agent_steps or (30 if a.benchmark=="official" else 14)
    a.temperature = a.temperature if a.temperature is not None else (1.0 if a.benchmark=="official" else 0.2)
    if a.num_trials<1 or a.max_agent_steps<1 or a.max_user_turns<0 or a.max_concurrency<1:
        p.error("参数次数无效")
    selected=list(ARMS) if a.arms=="all" else [x.strip() for x in a.arms.split(",")]
    if not selected or any(x not in ARMS for x in selected):p.error("--arms 需要合法选择")
    if a.benchmark == "official" and a.arms != "all":
        p.error("官方冻结实验需要完整六组消融；--arms 仅适用于 --benchmark micro")
    taskids=list(range(len(TASK_SPEC))) if a.task_ids is None else a.task_ids
    if not taskids or any(i<0 or i>=len(TASK_SPEC) for i in taskids) or len(taskids)!=len(set(taskids)):
        p.error("--task-ids 必须是 0 到 9 的不重复编号")
    cfg=backend(a)
    out=Path(a.output_dir) if a.output_dir else Path(__file__).resolve().parent/"results"/stamp()
    plan={"experiment":"2-4", "benchmark": ("official tau-bench airline" if a.benchmark=="official" else "self-contained controlled airline micro-benchmark"),
          "official_tau_bench": a.benchmark=="official",
          "note": ("真实 τ-bench 数据/环境/用户模拟器/客观分数，首次须获取固定源码" if a.benchmark=="official" else
                   "六组消融与客观判分可真实运行，但不是上游完整 τ-bench 数据、工具或分数。"),
          "provider":cfg["provider"],"model":cfg["model"],"endpoint":cfg["url"],"arms":selected,"task_ids":taskids,
          "num_trials":a.num_trials,"user_simulator":("official tau-bench llm" if a.benchmark=="official" else a.user_simulator),
          "expected_cells":len(selected)*len(taskids)*a.num_trials,"output":str(out)}
    if a.dry_run:
        print(json.dumps({**plan,"key_present":bool(cfg["key"])},ensure_ascii=False,indent=2));return
    if not cfg["key"]:p.error("未配置 DASHSCOPE_API_KEY/MOONSHOT_API_KEY。或使用 --provider local")
    if a.benchmark=="official":
        official_experiment(a,cfg,out,plan)
        return
    from openai import OpenAI
    from tqdm import tqdm
    client=OpenAI(api_key=cfg["key"],base_url=cfg["url"],timeout=a.timeout,max_retries=1)
    save(out/"protocol.json",plan)
    results={}
    with tqdm(total=plan["expected_cells"],desc="2-4 六组消融",unit="任务") as bar:
        for arm in selected:
            records=[]
            for taskid in taskids:
                for trial in range(1,a.num_trials+1):
                    row=run_one(client,cfg,arm,task(taskid),a)
                    row["trial"]=trial
                    records.append(row)
                    save(out/"cells"/f"{arm}_{taskid:02d}_{trial:02d}.json",row)
                    bar.set_postfix_str(f"{arm}/{taskid}: score={row['reward']}")
                    bar.update(1)
            accepted=[x for x in records if x["complete"] and x["reward"] is not None]
            results[arm]={"tasks":len(records),"valid_tasks":len(accepted),
                          "passes":sum(x["reward"] for x in accepted),
                          "pass_rate":sum(x["reward"] for x in accepted)/len(accepted) if accepted else None,
                          "avg_agent_steps":sum(x["agent_steps"] for x in accepted)/len(accepted) if accepted else None,
                          "api_calls":sum(x["api_calls"] for x in records),
                          "total_tokens":sum(x["total_tokens"] for x in records)}
            print("[消融]",arm,results[arm])
    complete=all(x["valid_tasks"]==x["tasks"] for x in results.values())
    save(out/"comparison.json",{"status":"complete" if complete else "partial",
                                "official_tau_bench":False,"benchmark":"micro",
                                "config":plan,"results":results})
    chart(results,out/"ablation_success_rates.png")
    artifact_hashes={}
    for f in sorted((out / "cells").glob("*.json")) + [out/"protocol.json", out/"comparison.json", out/"ablation_success_rates.png"]:
        if f.is_file():
            artifact_hashes[f.relative_to(out).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    save(out/"manifest.json",{"experiment":"2-4","benchmark":"micro","official_tau_bench":False,
                               "status":"complete" if complete else "partial",
                               "expected_tasks":plan["expected_cells"],
                               "valid_tasks":sum(x["valid_tasks"] for x in results.values()),
                               "artifacts_sha256":artifact_hashes})
    print("\n[最终结果]")
    for arm,x in results.items():
        print(f"{arm:<17} valid={x['valid_tasks']:>2}/{x['tasks']}  pass={x['passes']:>2}  rate={x['pass_rate']}")
    print("[输出]",out)
    print("[说明] 这不是官方 τ-bench 冻结实验成绩；如需论文级官方复现，须使用原仓库完整数据与环境。")
    if not complete:sys.exit(2)

if __name__=="__main__":main()
