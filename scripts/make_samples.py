"""Task 3.6: 10 visual samples per backend, same 3 test writers and same texts, -> results/<run_id>/samples/.

    python scripts/make_samples.py --backends real baseline glyph_vae --allow-test     # local
    python scripts/make_samples.py --backends vatr --allow-test                         # Colab (GPU)

The sample set is fixed ONCE in results/samples_spec.json (committed) and every later backend reuses it verbatim, so
all backends are compared like for like:
  * writers = the first 3 (sorted) of the protocol's seeded 20-writer test subset (evaluation.protocol.seed 1337);
  * references = that writer's K=1 protocol reference line (P.split_lines, same seed) -- the protocol's own draw;
  * texts = TEXTS below (10 fixed lines, lower-case letters and spaces as the protocol renders them, <= 40 chars,
    chosen to contain letters that are usually absent from one reference line: q, z, x, j, k, v);
  * sample i = writer W[i % 3] writes TEXTS[i].
The "real" backend cannot write arbitrary text: its sample i is writer W[i % 3]'s (i // 3)-th protocol held-out line
(the lines the generated samples are scored against), recorded in the manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.data.iam import load_iam
from src.evaluation import protocol as P
from src.evaluation.harness import git_info, truncate_text
from src.evaluation.protocol import Ref
from src.utils.config import load_config, resolve_path

TEXTS = [
    "the quick brown fox jumps over the dog",
    "pack my box with five dozen liquor jugs",
    "a lazy zebra quietly vexed the judge",
    "just keep walking toward the six hills",
    "we quickly amazed the jovial boxers",
    "sphinx of black quartz judge my vow",
    "the wizard quickly jinxed the gnomes",
    "a jury of six kept vexing the queen",
    "five boxing wizards jump quickly now",
    "my crazy uncle bought quite a few jets",
]
N_WRITERS, K, SEED = 3, 1, 1337
SPEC = "results/samples_spec.json"


def build_spec(ds, cfg) -> dict:
    pc, ev = cfg["evaluation"]["protocol"], cfg["evaluation"]
    subset = P.choose_writers(ds, pc["n_writers"], pc["seed"], max(pc["ks"]) + pc["targets_per_writer"] // 2 + 1)
    writers = subset[:N_WRITERS]
    per = {}
    for w in writers:
        samples = [s for s in ds.samples if s.writer_id == w]
        refs, targets = P.split_lines(samples, K, pc["targets_per_writer"], ev["min_reference_letters"], pc["seed"], w)
        per[w] = {"ref_ids": [r.sample_id for r in refs], "ref_texts": [r.transcription for r in refs],
                  "heldout_ids": [t.sample_id for t in targets]}
    texts = [truncate_text(t, ev["max_target_chars"]) for t in TEXTS]
    assert texts == TEXTS, "TEXTS must already be in the protocol's rendered form"
    items = [{"i": i, "writer": writers[i % N_WRITERS], "text": texts[i]} for i in range(len(texts))]
    return {"task": "3.6", "created_utc": datetime.now(timezone.utc).isoformat(), "created_git": git_info()["sha"],
            "rule": "first 3 (sorted) writers of the protocol's seeded 20-writer test subset; K=1 protocol reference "
                    "line; sample i = writer[i % 3] writes texts[i]; real backend: writer[i % 3]'s (i // 3)-th held-out line",
            "protocol_seed": pc["seed"], "K": K, "writers": writers, "per_writer": per, "texts": texts, "items": items}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if not args.allow_test:
        print("Refusing to read the test split without --allow-test.", file=sys.stderr)
        return 2
    cfg = load_config()
    ds = load_iam("test", unit="lines")
    index = {s.sample_id: i for i, s in enumerate(ds.samples)}
    spec_path = resolve_path(SPEC)
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert spec["texts"] == TEXTS, "samples_spec.json and TEXTS disagree"
    else:
        spec = build_spec(ds, cfg)
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    git = git_info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from src.evaluation.protocol_cli import _make_backend

    for name in args.backends:
        random.seed(SEED); np.random.seed(SEED)
        try:
            import torch
            torch.manual_seed(SEED)
        except ImportError:
            pass
        run_id = f"{stamp}_{git['sha']}{'-dirty' if git['dirty'] else ''}_samples_{name}"
        out = resolve_path("results") / run_id / "samples"
        out.mkdir(parents=True, exist_ok=False)
        backend = None if name == "real" else _make_backend(name, cfg)
        manifest = []
        for w in spec["writers"]:
            pw = spec["per_writer"][w]
            for j, rid in enumerate(pw["ref_ids"]):
                cv2.imwrite(str(out / f"ref_{w}_{j}.png"), ds.read_native(index[rid]))
        for it in spec["items"]:
            w, pw = it["writer"], spec["per_writer"][it["writer"]]
            fname = f"{it['i']:02d}_{w}.png"
            if backend is None:
                sid = pw["heldout_ids"][it["i"] // N_WRITERS]
                img, text = ds.read_native(index[sid]), ds.samples[index[sid]].transcription
                entry = {"file": fname, "writer": w, "real_line": sid, "text": text}
            else:
                refs = [Ref(ds.read_native(index[r]), t) for r, t in zip(pw["ref_ids"], pw["ref_texts"])]
                img, text = backend.generate(refs, it["text"]), it["text"]
                entry = {"file": fname, "writer": w, "text": text, "ref_ids": pw["ref_ids"]}
            ok = img is not None and bool(np.any(np.asarray(img) < 128))
            if img is not None:
                cv2.imwrite(str(out / fname), img)
            manifest.append({**entry, "ok": ok})
            print(f"{name} {fname} writer {w} ok={ok} :: {text[:50]}")
        (out.parent / "manifest.json").write_text(json.dumps(
            {"run_id": run_id, "backend": name, "git": git, "spec": SPEC, "writers": spec["writers"], "K": spec["K"],
             "seed": SEED, "items": manifest, "n_ok": sum(m["ok"] for m in manifest)}, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {sum(m['ok'] for m in manifest)}/{len(manifest)} samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
