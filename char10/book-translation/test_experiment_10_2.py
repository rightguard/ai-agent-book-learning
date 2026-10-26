#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline regression tests. No network and no paid provider calls."""
import contextlib
import hashlib
import io
import json
from argparse import Namespace
from unittest.mock import patch
import os
import sys
from types import SimpleNamespace
from pathlib import Path
import re
import tempfile
import unittest

import experiment_10_2 as ex


SRC_1 = """# Chapter 1: Some Concept

A token is used in a prompt. UniqueSecretSourceSentinel one.

```python
def f(x):
    return x + 1
```

Visit [documentation](https://example.org/help) for more.
"""
SRC_2 = """# Chapter 2: Other Concept

Attention and embedding are concepts. UniqueSecretSourceSentinel two.

```python
print('hello')
```

A second paragraph with latency and token.
"""


class FakeGateway:
    def __init__(self, *, break_first_translation=False, invalid_glossary=False,
                 invalid_glossary_once=False, invalid_manager_plan=False,
                 judge_invalid=False, judge_fake_omission=False,
                 unresolved_review=False, missing_usage=False,
                 global_unresolved=False, global_invalid=False):
        self.calls = []
        self.break_first_translation = break_first_translation
        self.invalid_glossary = invalid_glossary
        self.invalid_glossary_once = invalid_glossary_once
        self.glossary_attempts = 0
        self.invalid_manager_plan = invalid_manager_plan
        self.judge_invalid = judge_invalid
        self.judge_fake_omission = judge_fake_omission
        self.unresolved_review = unresolved_review
        self.missing_usage = missing_usage
        self.global_unresolved = global_unresolved
        self.global_invalid = global_invalid

    def request(self, messages, max_tokens):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Glossary Agent" in system:
            self.glossary_attempts += 1
            bad = self.invalid_glossary or (self.invalid_glossary_once and self.glossary_attempts == 1)
            content = "nonsense" if bad else json.dumps({"terms": {"attention": "注意力"}})
        elif "Manager Agent" in system and "执行顺序" in system:
            ids = re.findall(r'"file":\s*"([^"]+)"', user)
            content = json.dumps({"order": ids[:-1] if self.invalid_manager_plan else ids})
        elif "Manager Agent" in system:
            need = '"needs_revision": true' in user or '"fenced_code_exact": false' in user
            content = json.dumps({"revise": need, "reason": "按验收结果返工"})
        elif "Global Proofreading Agent" in system:
            names = re.findall(r'<DOCUMENT filename="([^"]+)">', user)
            if self.global_invalid:
                content = '{"issues_by_file": {}}'
            else:
                issues_by_file = {name: [] for name in names}
                if self.global_unresolved and names:
                    issues_by_file[names[0]] = [{"severity": "high", "category": "term_drift",
                                                 "description": "跨章节术语不一致，请修订",
                                                 "evidence": "中文技术段落"}]
                content = json.dumps({"issues_by_file": issues_by_file,
                                      "summary": "跨章节检查完成"}, ensure_ascii=False)
        elif "Proofreading Agent" in system:
            need = self.unresolved_review or '"fenced_code_exact": false' in user
            content = json.dumps({"needs_revision": need,
                                  "issues": [{"severity": "high", "category": "structure",
                                              "description": "代码块应逐字保持"}] if need else [],
                                  "summary": "结构问题" if need else "未发现问题"}, ensure_ascii=False)
        elif "实验结束后的翻译质量评价者" in system:
            if self.judge_invalid:
                content = json.dumps({"accuracy": "bad", "coverage": 4})
            else:
                content = json.dumps({"accuracy": 4, "coverage": 4, "fluency": 5,
                                      "terminology": 4,
                                      "omissions": [{"source_excerpt": "NeverPresentFact999",
                                                     "description": "凭空捏造的遗漏"}]
                                      if self.judge_fake_omission else [],
                                      "summary": "盲评完成"}, ensure_ascii=False)
        else:
            m = re.search(r"<SOURCE>\n(.*?)\n</SOURCE>", user, re.S)
            if not m:
                raise AssertionError("translate request missing SOURCE tag")
            source = m.group(1)
            out = []
            in_code = False
            for line in source.splitlines():
                if line.startswith("```"):
                    in_code = not in_code
                    out.append(line)
                elif in_code or not line.strip():
                    out.append(line)
                elif line.startswith("#"):
                    prefix = line.split(" ")[0]
                    out.append(f"{prefix} 第一章/第二章 模拟中文技术译文")
                else:
                    out.append("这是完整的中文技术段落：词元、提示词、嵌入向量、注意力、时延。" + line)
            content = "\n".join(out) + "\n"
            if self.break_first_translation and "修订" not in user and "Single" not in system:
                # Intentionally introduce a missing fence to exercise review+revision.
                content = re.sub(r"```python\n.*?```\n?", "", content, flags=re.S)
        return ex.Response(content=content,
                           prompt_tokens=None if self.missing_usage else 90,
                           completion_tokens=None if self.missing_usage else 45,
                           cached_tokens=10, latency_s=0.01, response_id="fake-test", finish_reason="stop")


class ParsingAndVerificationTests(unittest.TestCase):
    def test_json_plain(self):
        self.assertEqual(ex.parse_json_object('{"ok":true}'), {"ok": True})

    def test_json_markdown(self):
        self.assertEqual(ex.parse_json_object('```json\n{"a":1}\n```'), {"a": 1})

    def test_json_with_surrounding_prose(self):
        self.assertEqual(ex.parse_json_object('Sure {"a":1} end'), {"a": 1})

    def test_json_invalid_rejected(self):
        with self.assertRaises(ValueError):
            ex.parse_json_object('not-json-{broken}-')

    def test_p95_requires_five_trials(self):
        self.assertIsNone(ex.pctl([1, 2, 3], 95))
        self.assertEqual(ex.pctl([1, 2, 3, 4, 5], 95), 4.8)

    def test_markdown_exact_structure(self):
        target = SRC_1.replace("# Chapter 1: Some Concept", "# 第一章 概念") \
            .replace("A token is used in a prompt. UniqueSecretSourceSentinel one.",
                     "这是关于词元和提示词的一段完整中文内容。") \
            .replace("Visit [documentation](https://example.org/help) for more.",
                     "访问[文档](https://example.org/help)了解更多中文信息。")
        result = ex.verify_markdown(SRC_1, target, ex.HOUSE_TERMS)
        self.assertTrue(result["structural_pass"])
        self.assertEqual(result["glossary_term_doc_hits"], 2)

    def test_code_change_caught(self):
        result = ex.verify_markdown(SRC_1, SRC_1.replace("return x + 1", "return x + 2"), ex.HOUSE_TERMS)
        self.assertFalse(result["checks"]["fenced_code_exact"])

    def test_code_missing_caught(self):
        result = ex.verify_markdown(SRC_1, SRC_1.replace("```python\ndef f(x):\n    return x + 1\n```", ""), ex.HOUSE_TERMS)
        self.assertFalse(result["structural_pass"])

    def test_heading_level_caught(self):
        result = ex.verify_markdown(SRC_1, SRC_1.replace("# Chapter", "## Chapter") + " 中文译文", ex.HOUSE_TERMS)
        self.assertFalse(result["checks"]["heading_levels_exact"])

    def test_link_target_caught(self):
        result = ex.verify_markdown(SRC_1, SRC_1.replace("https://example.org/help", "https://example.org/bad"), ex.HOUSE_TERMS)
        self.assertFalse(result["checks"]["link_targets_exact"])

    def test_inline_code_change_caught(self):
        source = "# Code\n\nCall `model.forward(x)` to run an inference.\n"
        target = "# 代码\n\n调用 `model.forward(y)` 完成推理。\n"
        result = ex.verify_markdown(source, target, ex.HOUSE_TERMS)
        self.assertFalse(result["checks"]["inline_code_exact"])

    def test_reference_link_target_change_caught(self):
        source = "# Refs\n\nVisit [docs][guide].\n\n[guide]: https://example.org/guide\n"
        target = "# 引用\n\n阅读[文档][guide]。\n\n[guide]: https://evil.invalid/guide\n"
        result = ex.verify_markdown(source, target, ex.HOUSE_TERMS)
        self.assertFalse(result["checks"]["reference_targets_exact"])

    def test_link_inside_code_not_counted(self):
        original = "# X\n\n```python\nprint('[a](http://source)')\n```\n"
        result = ex.verify_markdown(original, original.replace("# X", "# 中文标题") + "中文说明\n", {})
        self.assertTrue(result["checks"]["link_targets_exact"])

    def test_images_caught(self):
        orig = "# X\n\n![img](assets/a.png)\n"
        new = "# 中文标题\n\n![图](assets/b.png)\n中文内容翻译翻译\n"
        result = ex.verify_markdown(orig, new, {})
        self.assertFalse(result["checks"]["image_targets_exact"])

    def test_paragraph_proxy_not_hard_gate(self):
        orig = "# Heading\n\nFirst paragraph.\n\nSecond paragraph.\n"
        target = "# 标题\n\n第一段和第二段合并形成一段完整中文内容。\n"
        result = ex.verify_markdown(orig, target, {})
        self.assertTrue(result["structural_pass"])
        self.assertLess(result["paragraph_coverage_proxy"], 1.0)

    def test_glossary_does_not_count_code_only(self):
        orig = "# Heading\n\n```python\ntoken = 1\n```\n"
        translated = "# 中文标题\n\n```python\ntoken = 1\n```\n中文翻译文本。"
        result = ex.verify_markdown(orig, translated, ex.HOUSE_TERMS)
        self.assertNotIn("token", result["term_details"])

    def test_doc_load_sort_and_limit(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "chapter2.md").write_text("## two", encoding="utf-8")
            (p / "chapter1.md").write_text("## one", encoding="utf-8")
            docs = ex.load_documents(p, 1)
            self.assertEqual([d.name for d in docs], ["chapter1.md"])

    def test_file_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            with self.assertRaises(FileNotFoundError):
                ex.load_documents(p, 2)
            (p / "empty.md").write_text("   ", encoding="utf-8")
            with self.assertRaises(ValueError):
                ex.load_documents(p, 2)

    def test_meter_missing_usage(self):
        m = ex.Meter()
        m.record("manager_plan", ex.Response("x", None, None, None, 0.1))
        self.assertEqual(m.calls_missing_usage, 1)
        self.assertIsNone(m.manager_input_peak)
        self.assertIsNone(m.to_dict()["input_tokens"])
        self.assertIsNone(m.to_dict()["output_tokens"])
        self.assertIsNone(m.to_dict()["total_tokens"])
        self.assertIsNone(m.to_dict()["by_role"]["manager_plan"]["total_tokens"])

    def test_mixed_usage_does_not_silently_count_missing_as_zero(self):
        m = ex.Meter()
        m.record("single", ex.Response("a", 50, 20, 0, 0.1))
        m.record("single", ex.Response("b", None, 30, None, 0.1))
        self.assertIsNone(m.to_dict()["input_tokens"])
        self.assertEqual(m.to_dict()["output_tokens"], 50)
        self.assertIsNone(m.to_dict()["total_tokens"])
        self.assertEqual(m.to_dict()["observed_input_tokens"], 50)

    def test_optional_mean_incomplete(self):
        self.assertIsNone(ex.optional_mean([100, None, 200]))
        self.assertIsNone(ex.optional_mean([]))
        self.assertEqual(ex.optional_mean([100, 200]), 150)

    def test_custom_host_does_not_receive_dashscope_key_by_default(self):
        args = Namespace(base_url="https://other-provider.example/v1", api_key_env=None)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "secret-dashscope-key", "OPENAI_API_KEY": ""}):
            with self.assertRaises(ValueError):
                ex.find_api_key_and_base(args)
            args.api_key_env = "DASHSCOPE_API_KEY"  # Explicit override is allowed.
            key, _, name = ex.find_api_key_and_base(args)
            self.assertEqual((key, name), ("secret-dashscope-key", "DASHSCOPE_API_KEY"))

    def test_api_url_must_be_https_and_have_no_embedded_secrets(self):
        for base in ("http://other-provider.example/v1",
                     "https://user:password@other-provider.example/v1",
                     "https://other-provider.example/v1?key=secret"):
            with self.subTest(base=base):
                args = Namespace(base_url=base, api_key_env="OPENAI_API_KEY")
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
                    with self.assertRaises(ValueError):
                        ex.find_api_key_and_base(args)

    def test_thinking_default_matches_provider(self):
        args = Namespace(model="qwen3.7-flash-2026-07-15", thinking="auto", disable_thinking=False)
        self.assertEqual(ex.resolve_thinking_mode(args, "https://dashscope.aliyuncs.com/compatible-mode/v1"), "off")
        self.assertEqual(ex.resolve_thinking_mode(args, "https://example.org/v1"), "auto")
        args.thinking = "on"
        self.assertEqual(ex.resolve_thinking_mode(args, "https://dashscope.aliyuncs.com/compatible-mode/v1"), "on")
        args.disable_thinking = True
        with self.assertRaises(ValueError):
            ex.resolve_thinking_mode(args, "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_shared_call_budget_cannot_be_exceeded(self):
        fake = FakeGateway()
        budget = ex.CallBudget(limit=2)
        a = ex.BoundedGateway(fake, budget)
        b = ex.BoundedGateway(fake, budget)
        a.request(ex.make_messages("你是 Glossary Agent", "source"), 100)
        b.request(ex.make_messages("你是 Glossary Agent", "source"), 100)
        with self.assertRaises(ex.BudgetExhausted):
            a.request(ex.make_messages("你是 Glossary Agent", "source"), 100)
        self.assertEqual(budget.used, 2)
        self.assertEqual(len(fake.calls), 2)

    def test_budget_block_is_not_network_request_error(self):
        with tempfile.TemporaryDirectory() as td:
            blocked = ex.BoundedGateway(FakeGateway(), ex.CallBudget(limit=0))
            recorder = ex.StageRecorder(Path(td), blocked, "fake", 0, 500, 500)
            with self.assertRaises(ex.BudgetExhausted):
                recorder.ask("glossary", ex.make_messages("你是 Glossary Agent", "source"),
                             json_output=True)
            self.assertEqual(recorder.meter.request_errors, 0)
            self.assertEqual(recorder.meter.calls, 0)
            self.assertIn('"type": "budget_blocked"',
                          (Path(td) / "agent_trace.jsonl").read_text(encoding="utf-8"))

    def test_openai_compatible_api_gateway_arguments_and_usage(self):
        captured = {}
        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    id="api-response-1",
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="translated content"),
                        finish_reason="stop")],
                    usage=SimpleNamespace(
                        prompt_tokens=120, completion_tokens=80,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=15)))
        class FakeOpenAIClient:
            def __init__(self, **kwargs):
                captured["constructor"] = kwargs
                self.chat = SimpleNamespace(completions=Completions())
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=FakeOpenAIClient)}):
            gateway = ex.APIGateway(api_key="test-key", base_url="https://example.org/v1",
                                    model="qwen-test", temperature=0.0, timeout_s=45,
                                    thinking_mode="off")
            resp = gateway.request(ex.make_messages("system", "user"), max_tokens=500)
        self.assertEqual(captured["constructor"]["max_retries"], 2)
        self.assertEqual(captured["extra_body"], {"enable_thinking": False})
        self.assertEqual(captured["max_tokens"], 500)
        self.assertEqual(resp.content, "translated content")
        self.assertEqual((resp.prompt_tokens, resp.completion_tokens, resp.cached_tokens),
                         (120, 80, 15))
        self.assertEqual(resp.response_id, "api-response-1")

    def test_glossary_large_source_is_chunked_not_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            docs = [ex.Document("chapter1.md", "token " * 2200, "abc", p / "chapter1.md"),
                    ex.Document("chapter2.md", "SpecialTailMarker " + "attention " * 1700,
                                "def", p / "chapter2.md")]
            fake = FakeGateway()
            recorder = ex.StageRecorder(p / "shared", fake, "fake", 0, 1000, 300)
            _, info = ex.build_glossary(docs, recorder)
            self.assertEqual(info["chunks"], 2)
            self.assertEqual(recorder.meter.calls, 2)
            glossary_calls = ["\n".join(m["content"] for m in c["messages"])
                              for c in fake.calls if "Glossary Agent" in c["messages"][0]["content"]]
            self.assertTrue(any("SpecialTailMarker" in s for s in glossary_calls))


class ExperimentEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = self.root / "sample_book"
        self.samples.mkdir()
        (self.samples / "chapter1.md").write_text(SRC_1, encoding="utf-8")
        (self.samples / "chapter2.md").write_text(SRC_2, encoding="utf-8")
        self.outputs = self.root / "results"

    def tearDown(self):
        self.temp.cleanup()

    def run_experiment(self, gateway, more=None):
        argv = ["--model", "fake-test-model", "--project-root", str(self.root),
                "--sample-dir", str(self.samples), "--output-root", str(self.outputs),
                "--max-chapters", "2", "--trials", "1", "--json-retries", "0"]
        if more:
            argv += more
        with contextlib.redirect_stdout(io.StringIO()):
            code = ex.main(argv, gateway_override=gateway)
        runs = list(self.outputs.glob("exp10_2_*"))
        self.assertEqual(len(runs), 1)
        return code, runs[0]

    def test_real_e2e_pipeline_with_fake_only(self):
        fake = FakeGateway()
        code, root = self.run_experiment(fake)
        self.assertEqual(code, 0)
        data = json.loads((root / "results.json").read_text(encoding="utf-8"))
        trial = data["trials"][0]
        self.assertEqual(set(trial["arms"]), {"single", "multi", "multi-review"})
        for arm in trial["arms"].values():
            self.assertEqual(arm["status"], "PASS")
            self.assertEqual(arm["evaluation"]["output_files"], 2)
            self.assertGreater(arm["meter"]["api_calls"], 0)
        self.assertEqual(trial["shared_meter"]["api_calls"], 1)
        self.assertGreater(trial["arms"]["multi"]["meter"]["manager_context_peak_tokens"], 0)
        self.assertTrue((root / "summary.md").exists())
        self.assertTrue((root / "manifest.json").exists())

    def test_manager_never_receives_original_full_text(self):
        fake = FakeGateway()
        self.run_experiment(fake, ["--mode", "multi-review"])
        manager_calls = [x for x in fake.calls if "Manager Agent" in x["messages"][0]["content"]]
        self.assertGreater(len(manager_calls), 0)
        for call in manager_calls:
            alltext = "\n".join(m["content"] for m in call["messages"])
            self.assertNotIn("UniqueSecretSourceSentinel", alltext)

    def test_absent_glossary_fallback(self):
        fake = FakeGateway(invalid_glossary=True)
        code, root = self.run_experiment(fake, ["--mode", "single"])
        self.assertEqual(code, 1)
        trial = json.loads((root / "trial_01" / "trial_results.json").read_text(encoding="utf-8"))
        self.assertTrue(trial["glossary"]["fallback"])
        self.assertEqual(trial["arms"]["single"]["status"], "NOT_PASS")
        self.assertTrue(trial["arms"]["single"]["structure_gate_pass"])
        self.assertFalse(trial["arms"]["single"]["protocol_gate_pass"])

    def test_skip_glossary_no_shared_api_call(self):
        code, root = self.run_experiment(FakeGateway(), ["--mode", "single", "--skip-glossary-llm"])
        self.assertEqual(code, 0)
        data = json.loads((root / "trial_01" / "trial_results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["shared_meter"]["api_calls"], 0)

    def test_dry_run_never_calls_api_or_creates_run(self):
        fake = FakeGateway()
        argv = ["--model", "fake-test-model", "--sample-dir", str(self.samples),
                "--output-root", str(self.outputs), "--dry-run"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = ex.main(argv, gateway_override=fake)
        self.assertEqual(code, 0)
        self.assertEqual(fake.calls, [])
        self.assertFalse(self.outputs.exists())

    def test_no_key_material_written_in_raw_calls(self):
        fake = FakeGateway()
        _, root = self.run_experiment(fake)
        data = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.json"))
        self.assertNotIn("DASHSCOPE_API_KEY=", data)
        self.assertNotIn('"api_key":', data)

    def test_independent_reviewer_can_trigger_repair(self):
        fake = FakeGateway(break_first_translation=True)
        code, root = self.run_experiment(fake, ["--mode", "multi-review"])
        self.assertEqual(code, 0)
        arm = json.loads((root / "trial_01" / "multi-review" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(arm["execution"]["reworks"], 2)
        self.assertEqual(arm["execution"]["reviews"], 4)
        self.assertEqual(arm["execution"]["review_issues_final"], 0)
        self.assertTrue(arm["evaluation"]["deterministic_structure_pass"])
        self.assertTrue((root / "trial_01" / "multi-review" / "proofreading" / "chapter1.md.round2.json").is_file())
        self.assertTrue((root / "trial_01" / "multi-review" / "revisions" / "chapter1.md.round0.md").is_file())
        self.assertTrue((root / "trial_01" / "multi-review" / "revisions" / "chapter1.md.round1.diff").is_file())
        self.assertEqual(arm["execution"]["global_reviews"], 2)

    def test_global_reviewer_sees_all_chapters_not_manager(self):
        fake = FakeGateway()
        code, root = self.run_experiment(fake, ["--mode", "multi-review"])
        self.assertEqual(code, 0)
        global_calls = [x for x in fake.calls if "Global Proofreading Agent" in x["messages"][0]["content"]]
        self.assertEqual(len(global_calls), 2)
        for call in global_calls:
            user = call["messages"][-1]["content"]
            self.assertIn('<DOCUMENT filename="chapter1.md">', user)
            self.assertIn('<DOCUMENT filename="chapter2.md">', user)
            self.assertNotIn("<SOURCE>", user)
        self.assertTrue((root / "trial_01" / "multi-review" / "proofreading" / "global.initial.json").is_file())
        self.assertTrue((root / "trial_01" / "multi-review" / "proofreading" / "global.final.json").is_file())

    def test_global_review_schema_failure_is_protocol_fail(self):
        code, root = self.run_experiment(FakeGateway(global_invalid=True),
                                         ["--mode", "multi-review"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "multi-review" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(arm["execution"]["invalid_global_reviews"], 2)
        self.assertIn("global_proofreading_schema_invalid", arm["protocol_errors"])

    def test_global_issue_can_trigger_rework_and_final_unresolved_failure(self):
        code, root = self.run_experiment(FakeGateway(global_unresolved=True),
                                         ["--mode", "multi-review"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "multi-review" / "results.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(arm["execution"]["reworks"], 1)
        self.assertIn("chapter1.md", arm["execution"]["unresolved_review_files"])
        self.assertIn("review_unresolved_after_budget", arm["protocol_errors"])

    def test_missing_review_cannot_silently_pass(self):
        code, root = self.run_experiment(FakeGateway(break_first_translation=True), ["--mode", "multi"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "multi" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(arm["status"], "NOT_PASS")
        self.assertFalse(arm["evaluation"]["deterministic_structure_pass"])

    def test_manager_plan_invalid_is_explicit_fallback(self):
        code, root = self.run_experiment(FakeGateway(invalid_manager_plan=True), ["--mode", "multi"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "multi" / "results.json").read_text(encoding="utf-8"))
        self.assertFalse(arm["execution"]["plan"]["valid"])
        self.assertTrue(arm["execution"]["plan"]["fallback"])
        self.assertIn("manager_plan_invalid", arm["protocol_errors"])

    def test_invalid_json_retries_and_costs_counted(self):
        fake = FakeGateway(invalid_glossary_once=True)
        code, root = self.run_experiment(fake, ["--mode", "single", "--json-retries", "1"])
        self.assertEqual(code, 0)
        trial = json.loads((root / "trial_01" / "trial_results.json").read_text(encoding="utf-8"))
        self.assertEqual(trial["shared_meter"]["api_calls"], 2)
        self.assertTrue(trial["glossary"]["json_valid"])
        self.assertEqual(fake.glossary_attempts, 2)

    def test_truncated_model_output_fails_and_preserves_trace(self):
        class TruncationGateway:
            def request(self, messages, max_tokens):
                return ex.Response("```json\n{}\n```", 30, 15, None, 0.1,
                                   "truncated", "length")
        with tempfile.TemporaryDirectory() as td:
            rec = ex.StageRecorder(Path(td), TruncationGateway(), "test", 0, 1000, 200)
            with self.assertRaises(RuntimeError):
                rec.ask("glossary", ex.make_messages("system", "user"), json_output=True)
            self.assertEqual(rec.meter.calls, 1)
            self.assertEqual(rec.meter.request_errors, 0)
            self.assertEqual(len(list((Path(td) / "raw_calls").glob("*.json"))), 1)
            trace = (Path(td) / "agent_trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("model_incomplete", trace)

    def test_official_four_samples_git_blobs(self):
        official = {
            "chapter1.md": "57379950d0cab7b7530f806e9fa6c342f2d331be",
            "chapter2.md": "d91ce3f7d18b018ec57ecebf8165da49baa03893",
            "chapter3.md": "18dc4616381c32558150430e079ba4dfba2f6c73",
            "chapter4.md": "2cfe64e9cb699e1ccddf07b3fff2e5224be29b0f",
        }
        for filename, expected in official.items():
            body = (ex.SCRIPT_DIR / "sample_book" / filename).read_bytes()
            actual = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
            self.assertEqual(actual, expected, filename)

    def test_full_trace_and_shared_cost_separated(self):
        fake = FakeGateway()
        _, root = self.run_experiment(fake)
        trial = json.loads((root / "trial_01" / "trial_results.json").read_text(encoding="utf-8"))
        self.assertEqual(trial["shared_meter"]["total_tokens"], 135)
        self.assertEqual(trial["arms"]["single"]["meter"]["total_tokens"], 270)
        for mode in ex.ARM_LABELS:
            self.assertTrue((root / "trial_01" / mode / "agent_trace.jsonl").exists())

    def test_uniform_quality_judge_is_accounted_separately(self):
        fake = FakeGateway()
        code, root = self.run_experiment(fake)
        self.assertEqual(code, 0)
        trial = json.loads((root / "trial_01" / "trial_results.json").read_text(encoding="utf-8"))
        self.assertEqual(trial["quality_meter"]["api_calls"], 6)  # 2 docs x 3 arms
        for arm in trial["arms"].values():
            self.assertTrue(arm["quality"]["all_judgments_valid"])
            self.assertEqual(arm["quality"]["mean_quality_score"], 4.25)
            self.assertEqual(arm["quality"]["reported_omissions"], 0)
        score_files = list((root / "trial_01" / "evaluation" / "scores").glob("*.json"))
        self.assertEqual(len(score_files), 6)
        self.assertIn("统一盲评成本", (root / "summary.md").read_text(encoding="utf-8"))
        self.assertIn("分角色调用指标", (root / "summary.md").read_text(encoding="utf-8"))
        for mode, arm in trial["arms"].items():
            by_role = arm["meter"]["by_role"]
            self.assertEqual(sum(v["api_calls"] for v in by_role.values()), arm["meter"]["api_calls"])
        budget = json.loads((root / "call_budget.json").read_text(encoding="utf-8"))
        self.assertEqual(budget["sdk_requests_reserved"], len(fake.calls))

    def test_false_omission_quote_fails_quality_protocol(self):
        code, root = self.run_experiment(FakeGateway(judge_fake_omission=True), ["--mode", "single"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "single" / "results.json").read_text(encoding="utf-8"))
        self.assertIn("final_quality_judge_invalid", arm["protocol_errors"])
        self.assertFalse(arm["quality"]["all_judgments_valid"])

    def test_invalid_quality_judge_never_pretends_to_pass(self):
        code, root = self.run_experiment(FakeGateway(judge_invalid=True), ["--mode", "single"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "single" / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(arm["structure_gate_pass"])
        self.assertFalse(arm["protocol_gate_pass"])

    def test_explicit_no_judge_is_labeled_not_measured(self):
        code, root = self.run_experiment(FakeGateway(judge_invalid=True),
                                         ["--mode", "single", "--no-judge"])
        self.assertEqual(code, 0)
        arm = json.loads((root / "trial_01" / "single" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(arm["quality"]["status"], "NOT_MEASURED")

    def test_unresolved_review_not_counted_as_success(self):
        code, root = self.run_experiment(FakeGateway(unresolved_review=True),
                                         ["--mode", "multi-review", "--max-revisions", "1"])
        self.assertEqual(code, 1)
        arm = json.loads((root / "trial_01" / "multi-review" / "results.json").read_text(encoding="utf-8"))
        self.assertTrue(arm["structure_gate_pass"])
        self.assertFalse(arm["execution"]["review_resolved"])
        self.assertIn("review_unresolved_after_budget", arm["protocol_errors"])

    def test_missing_usage_is_na_across_output_and_summary(self):
        code, root = self.run_experiment(FakeGateway(missing_usage=True), ["--mode", "single"])
        self.assertEqual(code, 0)
        arm = json.loads((root / "trial_01" / "single" / "results.json").read_text(encoding="utf-8"))
        self.assertIsNone(arm["meter"]["total_tokens"])
        self.assertGreater(arm["meter"]["calls_missing_usage"], 0)
        self.assertIn("N/A", (root / "summary.md").read_text(encoding="utf-8"))

    def test_five_trial_interleaving_and_percentile(self):
        gateway = FakeGateway()
        code, root = self.run_experiment(gateway, ["--trials", "5", "--no-judge"])
        self.assertEqual(code, 0)
        data = json.loads((root / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["trials"]), 5)
        orders = [x["execution_order"] for x in data["trials"]]
        self.assertNotEqual(orders[0], orders[1])
        self.assertNotEqual(orders[1], orders[2])
        summary = (root / "summary.md").read_text(encoding="utf-8")
        self.assertRegex(summary, r"\| A 单 Agent \| 5/5 \|")
        self.assertIn("p95", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
