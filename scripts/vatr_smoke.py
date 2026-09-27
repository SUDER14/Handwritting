"""Single-writer / single-K smoke test of the VATr backend (Task 3.3): load the released checkpoint, take K=1 reference
line of the first protocol test writer, generate the text of one of that writer's other lines, save PNGs + a JSON.

    python scripts/vatr_smoke.py [--device cpu|cuda] [--out results/vatr_smoke]

Test-split data is only READ here (no training), exactly as the protocol does; --allow-test is implied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.data.iam import load_iam
from src.evaluation import protocol as P
from src.evaluation.harness import git_info
from src.evaluation.protocol import Ref
from src.generator.vatr_backend import VATrBackend, reference_word_crops, transcription_words
from src.utils.config import load_config, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/vatr_smoke")
    args = ap.parse_args()
    cfg = load_config()
    pc = cfg["evaluation"]["protocol"]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_iam("test", unit="lines")
    writer = P.choose_writers(ds, pc["n_writers"], pc["seed"], 2)[0]
    idx = [i for i, s in enumerate(ds.samples) if s.writer_id == writer]
    ref_s, tgt_s = ds.samples[idx[0]], ds.samples[idx[1]]
    ref = Ref(ds.read_native(idx[0]), ref_s.transcription)

    t0 = time.time()
    backend = VATrBackend(cfg, device=args.device)
    t_load = time.time() - t0
    crops = reference_word_crops([ref])
    for i, c in enumerate(crops):
        cv2.imwrite(str(out / f"ref_word_{i:02d}.png"), c)
    t1 = time.time()
    img = backend.generate([ref], tgt_s.transcription)
    t_gen = time.time() - t1
    ok = img is not None and img.size > 0 and img.min() < 128 < img.max()
    if img is not None:
        cv2.imwrite(str(out / "generated.png"), img)
    cv2.imwrite(str(out / "reference.png"), ref.gray)
    rec = {"task": "3.3 smoke", "utc": datetime.now(timezone.utc).isoformat(), "git": git_info(),
           "device": backend.device, "checkpoint": str(backend.checkpoint), "writer": writer, "K": 1,
           "reference_line": ref_s.sample_id, "reference_text": ref_s.transcription,
           "n_transcription_words": len(transcription_words(ref_s.transcription)), "n_word_crops": len(crops),
           "target_line": tgt_s.sample_id, "target_text": tgt_s.transcription,
           "generated_shape": None if img is None else list(img.shape), "has_ink_and_paper": bool(ok),
           "load_s": round(t_load, 1), "generate_s": round(t_gen, 2), **backend.coverage()}
    (out / "smoke.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
