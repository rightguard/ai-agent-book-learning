# Experiment 5 Mini: Code-for-Logic x Coding Agent

- Model: `qwen3_0.6b` (local vLLM)
- Task: repair ONE wrong Boolean constraint in an isolated three-person Knights & Knaves solver
- This is a custom micro-experiment, not the official 84-puzzle benchmark.
- Source edit is restricted to one Boolean expression; test script is immutable.

## Outcomes

| Arm | Successful | Tokens | Time (s) |
|---|---|---:|---:|
| Pure reasoning | FAIL | 184 | 0.338 |
| Coding Agent (Read/Edit/Test/Finish) | FAIL | 8675 | 1.590 |

## Coding Agent execution

- Source reads: 1
- Source edits: 1
- Model-requested test runs: 3
- Final tests passed: 1/4
- Changed constraint: `a == b` -> `(a == b)`
- Final acceptance: FAIL

## Tool trajectory

- Step 1: `run_tests`; success=False
- Step 2: `read_file`; success=True
- Step 3: `edit_file`; success=True
- Step 4: `run_tests`; success=False
- Step 5: `run_tests`; success=False
- Step 6: `run_tests`; success=False

## Acceptance checks

| Check | Result |
|---|---|
| `no_runtime_error` | PASS |
| `vllm_model_available` | PASS |
| `pure_answer_valid` | PASS |
| `pure_answer_correct` | NOT_PASS |
| `source_read_by_agent` | PASS |
| `source_edited_by_agent` | PASS |
| `tests_invoked_by_agent` | PASS |
| `four_independent_tests_pass` | NOT_PASS |
| `agent_observed_pass_and_finished` | NOT_PASS |

## Boundaries

- No external API or real user repository edits.
- A scripted mock model is used only for --self-test, never for scored live runs.
- Pure reasoning and code repair measure distinct tasks; do not interpret their outcomes as a paired accuracy improvement.
- This micro-benchmark is too small to establish a general accuracy gain.
