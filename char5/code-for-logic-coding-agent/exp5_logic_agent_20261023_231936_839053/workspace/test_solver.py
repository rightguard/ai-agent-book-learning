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
