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
        c_says_true = (a == b)
        if a == a_says_true and b == b_says_true and c == c_says_true:
            solutions.append({
                "A": "knight" if a else "knave",
                "B": "knight" if b else "knave",
                "C": "knight" if c else "knave",
            })
    return solutions
