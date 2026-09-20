"""Edit-distance text metrics (no dependencies)."""
from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Minimum number of single-character insertions, deletions and substitutions turning `a` into `b`."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate = edit distance / len(reference). Can exceed 1.0 (many insertions)."""
    if not reference:
        raise ValueError("CER is undefined for an empty reference string")
    return levenshtein(reference, hypothesis) / len(reference)
