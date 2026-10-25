"""ROADMAP Task 5.4 [HUMAN]: build results/human_check/ -- 20 (generated vs real reference) pairs for the user to rate.

Each pair = one line image (A) + one REAL reference line (B); the rater guesses whether A and B were written by the same
person (same_writer_guess: yes/no) and how sure they are (confidence: 1-5). Sources are the Task 3.6 samples
(results/*_samples_*, writers 172/173/179, K=1): 8 baseline, 8 glyph_vae and 4 REAL held-out lines as a control
(shows whether the rater can do the task at all). Half of each source's pairs use the writer's own reference, half a
different writer's. Order and source are shuffled (seeded) and hidden; the answer key goes to answer_key.csv, which must
not be opened before rating. VATr pairs are added after the Colab run (not here).
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "human_check"
SEED = 5400
PLAN = {"baseline": 8, "glyph_vae": 8, "real": 4}
WRITERS = ["172", "173", "179"]


def sample_dirs() -> dict[str, Path]:
    dirs = {}
    for name in PLAN:
        found = sorted((ROOT / "results").glob(f"*_samples_{name}"))
        if not found:
            raise SystemExit(f"no results/*_samples_{name} (Task 3.6)")
        dirs[name] = found[-1]
    return dirs


def main() -> int:
    rng = np.random.default_rng(SEED)
    dirs = sample_dirs()
    items = []
    for src, n in PLAN.items():
        idx = list(rng.permutation(10)[:n])                          # sample index i -> writer WRITERS[i % 3]
        for k, i in enumerate(idx):
            w = WRITERS[i % 3]
            same = k % 2 == 0                                        # half same-writer, half different-writer
            ref_w = w if same else str(rng.choice([x for x in WRITERS if x != w]))
            items.append((src, int(i), w, ref_w, same))
    order = rng.permutation(len(items))
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "pairs").mkdir(parents=True)
    key_rows, csv_rows = [], []
    for pid, j in enumerate(order, start=1):
        src, i, w, ref_w, same = items[j]
        shutil.copy2(dirs[src] / "samples" / f"{i:02d}_{w}.png", OUT / "pairs" / f"pair_{pid:02d}_A.png")
        shutil.copy2(dirs["real"] / "samples" / f"ref_{ref_w}_0.png", OUT / "pairs" / f"pair_{pid:02d}_B_ref.png")
        csv_rows.append({"pair_id": f"pair_{pid:02d}", "same_writer_guess": "", "confidence": ""})
        key_rows.append({"pair_id": f"pair_{pid:02d}", "A_source": src, "sample_index": i, "A_writer": w,
                         "B_ref_writer": ref_w, "truth_same_writer": "yes" if same else "no"})
    with (OUT / "human_check.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["pair_id", "same_writer_guess", "confidence"])
        wr.writeheader()
        wr.writerows(csv_rows)
    with (OUT / "answer_key.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        wr.writeheader()
        wr.writerows(key_rows)
    (OUT / "README.md").write_text(
        "# Human check (Task 5.4) -- STATUS: DEFERRED, waiting for your ratings\n\n"
        "For each `pairs/pair_NN_A.png` (a line of handwriting, real or generated -- you are not told which) and "
        "`pairs/pair_NN_B_ref.png` (a REAL reference line), fill `human_check.csv`:\n\n"
        "* `same_writer_guess`: `yes` if you think A is in the handwriting of whoever wrote B, else `no`\n"
        "* `confidence`: 1 (guess) to 5 (certain)\n\n"
        "20 pairs, 3 writers (172, 173, 179), K=1 samples from Task 3.6. **Do not open `answer_key.csv` before "
        "rating** (it lists the source of A and the truth). Mix: 8 baseline, 8 glyph_vae, 4 real held-out control lines; "
        "half the pairs are same-writer, half different-writer. VATr pairs will be added after the Colab run.\n",
        encoding="utf-8")
    print(f"wrote {len(csv_rows)} pairs to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
