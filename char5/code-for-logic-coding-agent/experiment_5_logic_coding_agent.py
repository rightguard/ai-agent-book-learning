#!/usr/bin/env python3
"""Chapter 5 mini lab: Code-for-Logic combined with a constrained Coding Agent.

Single file; Python 3.10+; OpenAI-compatible local vLLM endpoint.
No live source file outside the generated results directory is modified.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


VERSION = "1.1.0"
EXPECTED_TESTS = 4
MAX_HISTORY_TURNS = 6
MAX_FEEDBACK_CHARS = 1800
NAMES = ("A", "B", "C")
GOLD = {"A": "knave", "B": "knight", "C": "knight"}
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3_0.6b"
START_EXPR = "a == b"

SOLVER_TEMPLATE = '''\
"""Small K&K constraint solver. Only c_says_true may be edited by the Agent."""
from itertools import product


def solve():
    """Return all identity assignments satisfying every person's statement."""
    solutions = []
    for a, b, c in product((False, True), repeat=3):
        # True means knight (always truthful); False means knave (always lying).
        # A says: "B is a knave."
        a_says_true = not b
        # B says: "C is a knight."
        b_says_true = c
        # C says: "A and B have different identities."
        # Repair this Boolean expression and verify the behavior.
        c_says_true = ({expression})
        if a == a_says_true and b == b_says_true and c == c_says_true:
            solutions.append({{
                "A": "knight" if a else "knave",
                "B": "knight" if b else "knave",
                "C": "knight" if c else "knave",
            }})
    return solutions
'''

TEST_SOURCE = '''\
"""Independent acceptance tests; this file is read-only for the Agent."""
import unittest
from solver import solve


class LogicTests(unittest.TestCase):
    def test_unique_solution(self):
        self.assertEqual(len(solve()), 1)

    def test_expected_identity(self):
        self.assertEqual(solve(), [{"A": "knave", "B": "knight", "C": "knight"}])

    def test_every_statement_is_respected(self):
        for row in solve():
            a = row["A"] == "knight"
            b = row["B"] == "knight"
            c = row["C"] == "knight"
            self.assertEqual(a, not b)
            self.assertEqual(b, c)
            self.assertEqual(c, a != b)

    def test_c_statement_truth_table(self):
        # A final solution alone cannot prove that the translated rule is valid.
        # The harness validates the editable AST before executing this test.
        import ast
        from itertools import product
        from pathlib import Path

        tree = ast.parse(Path(__file__).with_name("solver.py").read_text(encoding="utf-8"))
        rules = [node.value for node in ast.walk(tree)
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "c_says_true"]
        self.assertEqual(len(rules), 1)
        compiled = compile(ast.Expression(body=rules[0]), "<validated-boolean-rule>", "eval")
        for a, b, c in product((False, True), repeat=3):
            actual = eval(compiled, {"__builtins__": {}}, {"a": a, "b": b, "c": c})
            self.assertIs(type(actual), bool)
            self.assertEqual(actual, a != b, (a, b, c))


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

PROBLEM = (
    "Knights always tell the truth and knaves always lie. "
    "A says: B is a knave. B says: C is a knight. "
    "C says: A and B have different identities. "
    "Determine the unique identities of A, B and C.\n"
    "\u9a91\u58eb\u6c38\u8fdc\u8bf4\u771f\u8bdd\uff0c\u65e0\u8d56\u6c38\u8fdc\u8bf4\u5047\u8bdd\u3002A\u8bf4\uff1aB\u662f\u65e0\u8d56\u3002B\u8bf4\uff1aC\u662f\u9a91\u58eb\u3002C\u8bf4\uff1aA\u548cB\u8eab\u4efd\u4e0d\u540c\u3002\u8bf7\u63a8\u65ad\u6bcf\u4e2a\u4eba\u7684\u8eab\u4efd\u3002"
)

PURE_SYSTEM = (
    "Solve this Knights and Knaves logic puzzle yourself without tools. "
    "Output ONLY one JSON object with keys A, B and C. "
    'Each value must be "knight" or "knave". No prose.'
)

AGENT_SYSTEM = '''\
You are a SMALL Coding Agent working in a disposable Python exercise project.
The current solver.py has ONE deliberately incorrect logical expression.
Your task is to use evidence to fix the constraint and pass the tests.

Tools are exposed through this text protocol. Every response must contain
EXACTLY one JSON object (no markdown or explanation outside JSON):
  {"tool":"read_file","arguments":{"path":"solver.py"}}
  {"tool":"edit_file","arguments":{"path":"solver.py","new_expr":"BOOLEAN EXPRESSION"}}
  {"tool":"run_tests","arguments":{}}
  {"tool":"finish","arguments":{"answer":"brief summary"}}

Available tools: read_file, edit_file, run_tests, finish.
You may read solver.py and test_solver.py; you may only edit the expression
assigned to c_says_true in solver.py. It uses Python variables a, b and c,
where True is knight and False is knave. You cannot modify tests or any
other code. Make only the smallest necessary edit. Do not invent test results.

Required workflow: read solver.py -> edit_file -> run_tests -> finish.
If a test fails, read its feedback and revise the expression. At most one
JSON action per response. Do not repeatedly read/discover tools or finish
before a successful test run. Test success must be observed from the tool.
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def validate_expr(expr: str) -> str:
    expr = str(expr).strip()
    if len(expr) > 150:
        raise ValueError("expression too long (max 150 characters)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("expression is not valid Python: " + str(exc)) from exc
    allowed = (
        ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Name,
        ast.Load, ast.Eq, ast.NotEq, ast.Not, ast.And, ast.Or, ast.Constant,
        ast.BinOp, ast.BitXor,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("only boolean logic over a, b, c is allowed")
        if isinstance(node, ast.Name) and node.id not in ("a", "b", "c"):
            raise ValueError("unknown variable (only a, b, c allowed)")
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            raise ValueError("only boolean literals are allowed")
        if isinstance(node, ast.Compare) and len(node.ops) != 1:
            raise ValueError("comparison chains are not allowed")
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.BitXor):
            raise ValueError("only Boolean XOR (^) is allowed as a binary operator")
    return expr


def render_solver(expression: str) -> str:
    return SOLVER_TEMPLATE.format(expression=validate_expr(expression))


def init_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "solver.py").write_text(render_solver(START_EXPR), encoding="utf-8")
    (workspace / "test_solver.py").write_text(TEST_SOURCE, encoding="utf-8")


def expression_from_file(workspace: Path) -> str:
    source = (workspace / "solver.py").read_text(encoding="utf-8")
    match = re.search(r"^\s*c_says_true = \(([^\n]*)\)\s*$", source, re.MULTILINE)
    if not match:
        raise ValueError("missing editable expression")
    expr = validate_expr(match.group(1))
    if source != render_solver(expr):
        raise ValueError("source integrity violation: changes outside editable rule")
    if (workspace / "test_solver.py").read_text(encoding="utf-8") != TEST_SOURCE:
        raise ValueError("test integrity violation")
    return expr


def run_tests(workspace: Path) -> dict[str, Any]:
    try:
        expression_from_file(workspace)
    except Exception as exc:
        return {"ok": False, "error": "file_integrity_failure: " + str(exc), "tests_passed": 0, "tests_run": 0}
    started = time.perf_counter()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    def set_limits() -> None:
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
            resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))

    try:
        proc = subprocess.run(
            [sys.executable, "-B", "test_solver.py"],
            cwd=str(workspace), env=env, capture_output=True, text=True,
            timeout=8, check=False,
            preexec_fn=set_limits if sys.platform == "linux" else None,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        count = re.search(r"Ran\s+(\d+)\s+tests?", output)
        tests_run = int(count.group(1)) if count else 0
        failed_summary = re.search(r"FAILED\s*\(([^)]*)\)", output)
        failure_counts = [int(value) for kind, value in re.findall(
            r"(failures|errors|unexpected successes)=(\d+)",
            failed_summary.group(1) if failed_summary else "",
        )]
        n_failures = sum(failure_counts) if failed_summary else (0 if proc.returncode == 0 else tests_run)
        return {
            "ok": proc.returncode == 0 and tests_run == EXPECTED_TESTS,
            "exit_code": proc.returncode,
            "tests_run": tests_run,
            "tests_passed": max(0, tests_run - n_failures),
            "output": output[:10000],
            "duration_s": round(time.perf_counter() - started, 3),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "tests_timeout", "tests_run": 0, "tests_passed": 0}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__ + ": " + str(exc), "tests_run": 0, "tests_passed": 0}


def extract_json_object(content: str) -> Any:
    """Recover one action from text or a fenced JSON block, never two actions."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(content):
        if content[cursor] != "{":
            cursor += 1
            continue
        try:
            result, consumed = decoder.raw_decode(content[cursor:])
        except (json.JSONDecodeError, ValueError):
            cursor += 1
            continue
        if isinstance(result, dict):
            found.append(result)
            cursor += consumed
        else:
            cursor += 1
    return found[0] if len(found) == 1 else None


def parse_action(content: str) -> dict[str, Any]:
    obj = extract_json_object(content)
    if not isinstance(obj, dict):
        raise ValueError("expected exactly one JSON action object")
    tool = obj.get("tool") or obj.get("action")
    args = obj.get("arguments") if "arguments" in obj else obj.get("args", {})
    if tool == "call_tool":
        outer = args if isinstance(args, dict) else {}
        tool = obj.get("name") or outer.get("tool")
        args = outer.get("arguments", outer) if "tool" in outer else outer
    if not isinstance(tool, str):
        raise ValueError("missing tool name")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    if tool not in {"read_file", "edit_file", "run_tests", "finish"}:
        raise ValueError("unknown tool: " + tool)
    return {"tool": tool, "arguments": args}


def parse_pure_answer(content: str) -> Any:
    result = extract_json_object(content)
    if not isinstance(result, dict):
        return None
    normalized = {}
    for name in NAMES:
        val = str(result.get(name, "")).strip().lower()
        if val not in ("knight", "knave"):
            return None
        normalized[name] = val
    return normalized


def new_agent_state() -> dict[str, Any]:
    return {"edit_count": 0, "tests_count": 0, "tests_after_last_edit": False,
            "last_test": None, "finished": False, "read_solver_count": 0,
            "read_tests_count": 0}


def tool_observation(workspace: Path, action: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    name = action["tool"]
    args = action["arguments"]
    if name == "read_file":
        path = args.get("path", "solver.py")
        if not isinstance(path, str) or path not in {"solver.py", "test_solver.py"}:
            return {"ok": False, "error": "read restricted to solver.py and test_solver.py"}
        target = workspace / path
        try:
            expression_from_file(workspace)
            source = target.read_text(encoding="utf-8")
        except Exception as exc:
            return {"ok": False, "error": "file_integrity_failure: " + str(exc)}
        key = "read_solver_count" if path == "solver.py" else "read_tests_count"
        state[key] = state.get(key, 0) + 1
        return {"ok": True, "path": path, "content": source}

    if name == "edit_file":
        if state.get("read_solver_count", 0) < 1:
            return {"ok": False, "error": "read_first: inspect solver.py before editing"}
        if args.get("path", "solver.py") != "solver.py":
            return {"ok": False, "error": "only solver.py can be edited"}
        if "new_expr" not in args:
            return {"ok": False, "error": "provide arguments.new_expr with a Python boolean expression over a, b, c"}
        try:
            before = expression_from_file(workspace)
            new = str(args["new_expr"]).strip()
            if new.startswith("c_says_true") and "=" in new:
                new = new.split("=", 1)[1].strip()
            # Keep valid grouping parentheses intact; blindly dropping the
            # first and last parenthesis breaks expressions like (a != b) or (not c).
            new = validate_expr(new)
            if before == new:
                return {"ok": False, "error": "no change; pick a different expression", "expression": before}
            (workspace / "solver.py").write_text(render_solver(new), encoding="utf-8")
            state["edit_count"] += 1
            state["tests_after_last_edit"] = False
            state["last_test"] = None
            diff = "".join(difflib.unified_diff(
                render_solver(before).splitlines(keepends=True),
                render_solver(new).splitlines(keepends=True),
                fromfile="before/solver.py", tofile="after/solver.py",
            ))
            return {"ok": True, "old_expr": before, "new_expr": new, "diff": diff}
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}

    if name == "run_tests":
        report = run_tests(workspace)
        state["tests_count"] += 1
        state["last_test"] = report
        state["tests_after_last_edit"] = bool(report["ok"])
        return report

    if name == "finish":
        if (state.get("read_solver_count", 0) < 1 or state["edit_count"] < 1
                or state["tests_count"] < 1 or not state["tests_after_last_edit"]):
            return {"ok": False, "error": "finish rejected: read solver.py, edit it and observe passing run_tests"}
        state["finished"] = True
        return {"ok": True, "answer": args.get("answer", "")}
    raise RuntimeError("unreachable")


class VLLMChat:
    """Minimal OpenAI-compatible vLLM HTTP transport (Python standard library)."""

    def __init__(self, base_url: str, model: str, *, timeout: float, temperature: float,
                 max_tokens: int, disable_thinking: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        from urllib.parse import urlsplit
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username is not None or parsed.password is not None
                or parsed.path != "/v1" or parsed.query or parsed.fragment):
            raise ValueError("--base-url must be a local loopback HTTP endpoint ending exactly in /v1")

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, build_opener, ProxyHandler

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST" if data is not None else "GET",
        )
        try:
            # Never route a local vLLM call through an inherited HTTP proxy.
            opener = build_opener(ProxyHandler({}))
            with opener.open(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1600]
            raise RuntimeError("vLLM HTTP %s: %s" % (exc.code, details)) from exc
        except URLError as exc:
            raise ConnectionError(
                "Local vLLM unavailable at %s: %s" % (self.base_url, exc.reason)
            ) from exc

    def check(self) -> list[str]:
        payload = self._request("/models")
        names = [str(x.get("id")) for x in payload.get("data", [])]
        if self.model not in names:
            raise RuntimeError("model not offered by local server: " + self.model + "; models=" + repr(names))
        return names

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        params: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.disable_thinking:
            params["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.perf_counter()
        try:
            response = self._request("/chat/completions", params)
        except RuntimeError as exc:
            if self.disable_thinking and ("chat_template_kwargs" in str(exc) or "enable_thinking" in str(exc)):
                del params["chat_template_kwargs"]
                response = self._request("/chat/completions", params)
            else:
                raise
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM returned empty choices")
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        usage = response.get("usage") or {}
        return {
            "content": content,
            "reasoning_present": bool(msg.get("reasoning_content")),
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "latency_s": round(time.perf_counter() - started, 3),
        }


class ScriptedChat:
    """CI-only mock model; never used in real evaluation."""
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.replies:
            raise RuntimeError("mock replies exhausted")
        return {"content": self.replies.pop(0), "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_s": 0.0, "finish_reason": "stop"}


def aggregate_usage(traces: list[dict[str, Any]]) -> dict[str, int]:
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for trace in traces:
        for key in usage:
            value = trace.get("response", {}).get("usage", {}).get(key)
            if isinstance(value, int):
                usage[key] += value
    return usage


def run_pure(chat: Any) -> dict[str, Any]:
    started = time.perf_counter()
    response = chat.complete([{"role": "system", "content": PURE_SYSTEM},
                              {"role": "user", "content": PROBLEM}])
    prediction = parse_pure_answer(response["content"])
    return {"mode": "pure", "prediction": prediction, "correct": prediction == GOLD,
            "elapsed_s": round(time.perf_counter() - started, 3), "response": response,
            "expected": GOLD}


def run_agent(chat: Any, workspace: Path, max_steps: int = 10, verbose: bool = True) -> dict[str, Any]:
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": "Puzzle: " + PROBLEM +
                 "\nA test exists. Inspect the code, repair the single incorrect statement, run tests, and finish."}]
    trace: list[dict[str, Any]] = []
    state = new_agent_state()
    started = time.perf_counter()
    repeat_signature = None
    repeat_count = 0
    runtime_error = None
    try:
        from tqdm import tqdm
        iterator = tqdm(range(1, max_steps + 1), desc="Coding Agent", unit="step", disable=not verbose)
    except ImportError:
        iterator = range(1, max_steps + 1)
    for step in iterator:
        try:
            response = chat.complete(messages)
        except Exception as exc:
            runtime_error = type(exc).__name__ + ": " + str(exc)
            trace.append({"turn": step, "error_type": "model_runtime_failure", "error": runtime_error})
            break
        raw = str(response.get("content") or "")
        try:
            action = parse_action(raw)
            parse_error = None
        except ValueError as exc:
            action, parse_error = None, str(exc)
        if action is None:
            observation = {"ok": False, "error": "format_error: " + str(parse_error),
                           "hint": "Return exactly one JSON object containing tool and arguments."}
        else:
            signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
            repeat_count = repeat_count + 1 if signature == repeat_signature else 1
            repeat_signature = signature
            if repeat_count >= 3:
                observation = {"ok": False, "error": "repeated_same_action_three_times", "next": "STOP"}
            else:
                observation = tool_observation(workspace, action, state)
        entry = {"turn": step, "response": response, "raw_response_sha256": sha256(raw.encode()),
                 "action": action, "observation": observation}
        trace.append(entry)
        label = action["tool"] if action else "invalid_json"
        if verbose:
            message = "[Agent %02d/%02d] %-12s -> %s" % (step, max_steps, label, "PASS" if observation.get("ok") else "FAIL")
            if hasattr(iterator, "write"):
                iterator.write(message)
            else:
                print(message)
        messages.append({"role": "assistant", "content": raw})
        hint = (
            "Next action: read_file solver.py to inspect the code."
            if state["read_solver_count"] == 0
            else "Next action: edit_file to repair the incorrect c_says_true expression."
            if state["edit_count"] == 0
            else "Tests failed. Next action: edit_file with a revised expression, then run_tests again."
            if state["last_test"] is not None and not state["last_test"].get("ok")
            else "Next action: run_tests to verify your last edit."
            if not state["tests_after_last_edit"]
            else "All tests have passed. Next action: finish."
        )
        # Keep a bounded history for the 8K-token local model while preserving
        # the complete unabridged observation and model response in the trace.
        feedback = "Tool feedback: " + json.dumps(observation, ensure_ascii=False)[:MAX_FEEDBACK_CHARS]
        messages.append({"role": "user", "content": feedback +
                         "\n[STATE] reads=%d edits=%d tests=%d passing=%s\n" % (
                             state["read_solver_count"], state["edit_count"],
                             state["tests_count"], state["tests_after_last_edit"]) +
                         hint + "\nReturn exactly one JSON action."})
        if len(messages) > 2 + 2 * MAX_HISTORY_TURNS:
            messages = messages[:2] + messages[-2 * MAX_HISTORY_TURNS:]
        if observation.get("error") == "repeated_same_action_three_times" or state["finished"]:
            break
    check = run_tests(workspace)
    try:
        final_expr = expression_from_file(workspace)
        source_error = None
    except Exception as exc:
        final_expr = None
        source_error = type(exc).__name__ + ": " + str(exc)
    complete = bool(state["finished"] and state["read_solver_count"] > 0 and
                    state["edit_count"] > 0 and state["tests_count"] > 0 and
                    state["tests_after_last_edit"] and check.get("ok") and
                    not runtime_error and not source_error)
    return {"mode": "coding_agent", "complete": complete,
            "read_solver_count": state["read_solver_count"],
            "read_tests_count": state["read_tests_count"],
            "edits": state["edit_count"], "tests_invoked_by_agent": state["tests_count"],
            "tests_passed": check.get("tests_passed"), "final_test": check,
            "initial_expression": START_EXPR, "final_expression": final_expr,
            "runtime_error": runtime_error, "source_error": source_error,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "usage": aggregate_usage(trace), "trace": trace, "expected": GOLD}


def self_test() -> int:
    checks: list[tuple[str, bool]] = []
    def check(label: str, cond: bool) -> None:
        checks.append((label, bool(cond)))
        print("[%-4s] %s" % ("PASS" if cond else "FAIL", label))
    with tempfile.TemporaryDirectory(prefix="exp5_logic_") as tmp:
        ws = Path(tmp)
        init_workspace(ws)
        check("initial buggy solver fails independent tests", not run_tests(ws)["ok"])
        check("baseline source integrity", expression_from_file(ws) == START_EXPR)
        check("cannot import arbitrary modules", _expect_invalid("__import__('os').system('id')"))
        check("cannot call functions in boolean expression", _expect_invalid("str(a)"))
        check("cannot modify test_solver.py", not tool_observation(ws, {
            "tool": "edit_file", "arguments": {"path": "test_solver.py", "new_expr": "a != b"}
        }, {"edit_count": 0, "tests_after_last_edit": False})["ok"])
    with tempfile.TemporaryDirectory(prefix="exp5_agent_") as tmp:
        ws = Path(tmp)
        init_workspace(ws)
        fake = ScriptedChat([
            '{"tool":"read_file","arguments":{"path":"solver.py"}}',
            '{"tool":"edit_file","arguments":{"path":"solver.py","new_expr":"a != b"}}',
            '{"tool":"run_tests","arguments":{}}',
            '{"tool":"finish","arguments":{"answer":"tests passed"}}',
        ])
        result = run_agent(fake, ws, verbose=False)
        check("real file was edited", result["final_expression"] == "a != b")
        check("real Python unit tests passed", result["tests_passed"] == EXPECTED_TESTS)
        check("agent workflow accepted", result["complete"])
        check("exactly one successful edit", result["edits"] == 1)
        check("agent invoked tests", result["tests_invoked_by_agent"] == 1)
    with tempfile.TemporaryDirectory(prefix="exp5_finish_") as tmp:
        ws = Path(tmp)
        init_workspace(ws)
        state = new_agent_state()
        premature = tool_observation(ws, {"tool": "finish", "arguments": {}}, state)
        check("premature finish rejected", not premature["ok"] and not state["finished"])
    with tempfile.TemporaryDirectory(prefix="exp5_equivalence_") as tmp:
        ws = Path(tmp)
        init_workspace(ws)
        (ws / "solver.py").write_text(render_solver("(a != b) or c"), encoding="utf-8")
        check("semantic impostor rejected by truth-table test", not run_tests(ws)["ok"])
    with tempfile.TemporaryDirectory(prefix="exp5_read_gate_") as tmp:
        ws = Path(tmp)
        init_workspace(ws)
        state = new_agent_state()
        edit = tool_observation(ws, {"tool": "edit_file", "arguments": {"new_expr": "a != b"}}, state)
        check("editing without reading is rejected", not edit["ok"] and state["edit_count"] == 0)
    check("pure answer parser", parse_pure_answer('{"A":"knave","B":"knight","C":"knight"}') == GOLD)
    check("action JSON recovery", parse_action('```json\n{"tool":"run_tests","arguments":{}}\n```')["tool"] == "run_tests")
    passed = sum(value for _, value in checks)
    print("Self-test: %d/%d passed" % (passed, len(checks)))
    return 0 if passed == len(checks) else 1


def _expect_invalid(expression: str) -> bool:
    try:
        validate_expr(expression)
        return False
    except ValueError:
        return True


def find_root(script: Path) -> Path:
    for item in (script.parent, *script.parents):
        if (item / "config.yaml").exists() or (item / "qwen3_0.6b").is_dir():
            return item
    return Path.cwd()


def load_local_settings(root: Path) -> tuple[dict[str, Any], bool]:
    dotenv = root / ".env"
    loaded = False
    try:
        from dotenv import load_dotenv
        loaded = bool(load_dotenv(dotenv_path=dotenv, override=False)) if dotenv.exists() else False
    except ImportError:
        loaded = False
    cfg: dict[str, Any] = {}
    if (root / "config.yaml").exists():
        try:
            import yaml
            cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
            if not isinstance(cfg, dict):
                cfg = {}
        except ImportError:
            pass
    return cfg, loaded


def derive_acceptance(result: dict[str, Any]) -> dict[str, Any]:
    """Separate integrity/workflow evidence from the model's measured outcome."""
    results = result.get("results") or {}
    pure = results.get("pure")
    agent = results.get("coding_agent")
    checked = result.get("final_validation") or {}
    checks: dict[str, bool] = {
        "no_runtime_error": not bool(result.get("error")),
        "vllm_model_available": bool(result.get("model_ids")),
    }
    if pure is not None:
        checks["pure_answer_valid"] = pure.get("prediction") is not None
        checks["pure_answer_correct"] = bool(pure.get("correct"))
    if agent is not None:
        checks.update({
            "source_read_by_agent": agent.get("read_solver_count", 0) > 0,
            "source_edited_by_agent": agent.get("edits", 0) > 0,
            "tests_invoked_by_agent": agent.get("tests_invoked_by_agent", 0) > 0,
            "four_independent_tests_pass": bool(checked.get("ok")) and checked.get("tests_run") == EXPECTED_TESTS,
            "agent_observed_pass_and_finished": bool(agent.get("complete")),
        })
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "interpretation": "A recorded negative model outcome is a valid experiment result, not an infrastructure success",
    }


def save_manifest(out: Path) -> None:
    files = ["results.json", "summary.md", "agent_trace.jsonl", "changes.diff",
             "solver_before.py", "solver_after.py", "workspace/solver.py", "workspace/test_solver.py"]
    entries = {}
    for name in files:
        path = out / name
        if path.is_file():
            data = path.read_bytes()
            entries[name] = {"sha256": sha256(data), "bytes": len(data)}
    dump_json(out / "manifest.json", {"experiment": "chapter5_custom_logic_coding_agent",
                                      "version": VERSION, "files": entries})


def display_summary(out: Path, results: dict[str, Any], model: str,
                    acceptance: dict[str, Any] | None = None,
                    runtime_error: str | None = None) -> None:
    pure = results.get("pure")
    agent = results.get("coding_agent")
    lines = [
        "# Experiment 5 Mini: Code-for-Logic x Coding Agent",
        "",
        "- Model: `" + model + "` (local vLLM)",
        "- Task: repair ONE wrong Boolean constraint in an isolated three-person Knights & Knaves solver",
        "- This is a custom micro-experiment, not the official 84-puzzle benchmark.",
        "- Source edit is restricted to one Boolean expression; test script is immutable.",
        "",
        "## Outcomes",
        "",
        "| Arm | Successful | Tokens | Time (s) |",
        "|---|---|---:|---:|",
    ]
    if pure:
        u = pure["response"].get("usage") or {}
        lines.append("| Pure reasoning | %s | %s | %.3f |" %
                     ("PASS" if pure["correct"] else "FAIL", u.get("total_tokens", "N/A"), pure["elapsed_s"]))
    if agent:
        lines.append("| Coding Agent (Read/Edit/Test/Finish) | %s | %s | %.3f |" %
                     ("PASS" if agent["complete"] else "FAIL", agent["usage"].get("total_tokens", "N/A"), agent["elapsed_s"]))
        lines.extend(["", "## Coding Agent execution", "",
                      "- Source reads: %d" % agent["read_solver_count"],
                      "- Source edits: %d" % agent["edits"],
                      "- Model-requested test runs: %d" % agent["tests_invoked_by_agent"],
                      "- Final tests passed: %s/%s" % (agent["tests_passed"], EXPECTED_TESTS),
                      "- Changed constraint: `%s` -> `%s`" % (agent["initial_expression"], agent["final_expression"]),
                      "- Final acceptance: %s" % ("PASS" if agent["complete"] else "FAIL"),
                      "", "## Tool trajectory", ""])
        if agent.get("runtime_error"):
            lines.append("- Model runtime error: `%s`" % agent["runtime_error"])
        if agent.get("source_error"):
            lines.append("- Workspace integrity error: `%s`" % agent["source_error"])
        for row in agent["trace"]:
            action = row.get("action") or {}
            obs = row.get("observation") or {}
            lines.append("- Step %s: `%s`; success=%s" % (row["turn"], action.get("tool", "error"), bool(obs.get("ok"))))
    if acceptance:
        lines.extend(["", "## Acceptance checks", "", "| Check | Result |", "|---|---|"])
        for label, passed in acceptance["checks"].items():
            lines.append("| `%s` | %s |" % (label, "PASS" if passed else "NOT_PASS"))
    if runtime_error:
        lines.extend(["", "## Runtime failure", "", "`%s`" % runtime_error])
    lines.extend(["", "## Boundaries", "",
                  "- No external API or real user repository edits.",
                  "- A scripted mock model is used only for --self-test, never for scored live runs.",
                  "- Pure reasoning and code repair measure distinct tasks; do not interpret their outcomes as a paired accuracy improvement.",
        "- This micro-benchmark is too small to establish a general accuracy gain.", ""])
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Chapter 5 small integration: Code-for-Logic + Coding Agent + local vLLM")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--project-root", default=None)
    p.add_argument("--mode", choices=["both", "pure", "agent"], default="both")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if args.max_steps < 1 or args.max_steps > 30:
        p.error("--max-steps must be in [1, 30]")
    if args.max_tokens < 64 or args.max_tokens > 2048:
        p.error("--max-tokens must be in [64, 2048]")
    if not (0 <= args.temperature <= 2):
        p.error("--temperature must be in [0, 2]")
    if not (1 <= args.timeout <= 600):
        p.error("--timeout must be in [1, 600] seconds")
    script = Path(__file__).resolve()
    root = Path(args.project_root).resolve() if args.project_root else find_root(script)
    cfg, loaded = load_local_settings(root)
    local_cfg = cfg.get("local_vllm") or {}
    if not isinstance(local_cfg, dict):
        local_cfg = {}
    model = args.model or local_cfg.get("model") or os.getenv("LOCAL_VLLM_MODEL") or DEFAULT_MODEL
    base_url = args.base_url or local_cfg.get("base_url") or os.getenv("LOCAL_VLLM_BASE_URL") or DEFAULT_BASE_URL
    outbase = Path(args.output_dir).expanduser().resolve() if args.output_dir else script.parent / "results"
    print("Experiment 5 mini | Code-for-Logic + Coding Agent | v%s" % VERSION)
    print("Project root:", root)
    print("Config:", root / "config.yaml", "exists=", (root / "config.yaml").exists())
    print(".env:", root / ".env", "loaded=", loaded)
    print("Model:", model, "local model directory:", root / str(model))
    print("vLLM:", base_url, "key: EMPTY")
    print("Mode:", args.mode, "output root:", outbase)
    if args.dry_run:
        print("[PASS] Configuration dry run. No requests made and no workspace created.")
        return 0
    runid = "exp5_logic_agent_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = outbase / runid
    workspace = out / "workspace"
    init_workspace(workspace)
    (out / "solver_before.py").write_text(render_solver(START_EXPR), encoding="utf-8")
    result: dict[str, Any] = {
        "version": VERSION, "experiment": "chapter5_custom_logic_coding_agent",
        "model": model, "base_url": base_url,
        "official_benchmark": False, "workspace": str(workspace),
        "requested_mode": args.mode,
        "runtime": {"model": model, "base_url": base_url, "max_steps": args.max_steps,
                    "max_tokens": args.max_tokens, "temperature": args.temperature,
                    "timeout": args.timeout, "enable_thinking": args.enable_thinking},
        "puzzle": PROBLEM, "expected": GOLD, "results": {},
    }
    print("Output:", out)
    try:
        chat = VLLMChat(base_url, model, timeout=args.timeout, temperature=args.temperature,
                        max_tokens=args.max_tokens, disable_thinking=not args.enable_thinking)
        model_ids = chat.check()
        result["model_ids"] = model_ids
        print("[PASS] vLLM model check:", model_ids)
        if args.mode in {"both", "pure"}:
            print("[1/2] Pure reasoning")
            pure = run_pure(chat)
            result["results"]["pure"] = pure
            print("[%-4s] pure: %s" % ("PASS" if pure["correct"] else "FAIL", pure["prediction"]))
        if args.mode in {"both", "agent"}:
            print("[2/2] Constrained Coding Agent")
            agent = run_agent(chat, workspace, max_steps=args.max_steps)
            result["results"]["coding_agent"] = agent
            print("[%-4s] agent: reads=%s edits=%s test_runs=%s tests_passed=%s" %
                  ("PASS" if agent["complete"] else "FAIL", agent["read_solver_count"],
                   agent["edits"], agent["tests_invoked_by_agent"], agent["tests_passed"]))
            if agent.get("runtime_error"):
                result["error"] = "vLLM runtime failure: " + agent["runtime_error"]
    except Exception as exc:
        message = type(exc).__name__ + ": " + str(exc)
        result["error"] = message
        print("[FAIL]", message)
        if "Connection" in message or "ConnectError" in message:
            print("Hint: start local vLLM on port 8000 and check: curl http://127.0.0.1:8000/v1/models")
    finally:
        after = (workspace / "solver.py").read_text(encoding="utf-8")
        (out / "solver_after.py").write_text(after, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            render_solver(START_EXPR).splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="before/solver.py", tofile="after/solver.py",
        ))
        (out / "changes.diff").write_text(diff, encoding="utf-8")
        # A pure-reasoning run does not modify code: a broken exercise file is
        # expected there and must NOT be reported as a failed pure-run validation.
        result["final_validation"] = (run_tests(workspace)
                                      if args.mode in {"both", "agent"}
                                      else {"skipped": True, "reason": "pure mode has no code-editing arm"})
        result["acceptance"] = derive_acceptance(result)
        dump_json(out / "results.json", result)
        trace = (result.get("results", {}).get("coding_agent") or {}).get("trace") or []
        with (out / "agent_trace.jsonl").open("w", encoding="utf-8") as fp:
            for row in trace:
                fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        display_summary(out, result["results"], model, result["acceptance"], result.get("error"))
        save_manifest(out)
        print("Files: results.json, summary.md, agent_trace.jsonl, changes.diff, manifest.json, workspace/")
        if result["final_validation"].get("skipped"):
            print("Independent final validation: SKIPPED (pure reasoning has no code edits)")
        else:
            print("Independent final validation:", "PASS" if result["final_validation"].get("ok") else "NOT_PASS")
        print("Acceptance checks:")
        for label, passed in result["acceptance"]["checks"].items():
            print("  [%-8s] %s" % ("PASS" if passed else "NOT_PASS", label))
    if result.get("error"):
        return 2
    ok = all(x.get("correct") if name == "pure" else x.get("complete")
             for name, x in result["results"].items())
    print("Acceptance:", "PASS" if ok else "NOT_PASS (real model outcome)")
    return 0  # Experiment executed; a measured negative result is not a CLI crash.


if __name__ == "__main__":
    raise SystemExit(main())
