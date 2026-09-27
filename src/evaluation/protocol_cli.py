"""CLI body of `scripts/evaluate.py --mode protocol` (ROADMAP Task 1.4): run backends x K, write results/."""
from __future__ import annotations

import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data.iam import load_iam
from src.evaluation import protocol as P
from src.evaluation.harness import git_info
from src.evaluation.hwd_metric import HWDFeatures
from src.evaluation.writer_id import get_writer_embedder
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

TABLE_COLUMNS = ["run_id", "utc", "git_sha", "git_dirty", "split", "subset", "cpu_reduced", "backend", "K", "n_writers",
                 "cer", "wid_dist", "hwd", "oov_n_writers", "oov_cer", "oov_wid_dist", "oov_hwd",
                 "n_failed_generations", "crnn_run", "writer_id_run", "metrics_path"]


def _make_backend(name: str, cfg: dict):
    from src.evaluation import backends as B
    if name == "baseline":
        return B.BaselineBackend(cfg)
    if name == "glyph_vae":
        return B.NeuralBackend(cfg, str(resolve_path(cfg["training"]["checkpoint_path"])))
    if name == "vatr":
        from src.generator.vatr_backend import VATrBackend
        return VATrBackend(cfg)
    raise SystemExit(f"unknown backend {name!r}")


def run_protocol_cli(args) -> int:
    cfg = load_config()
    ev, pc = cfg["evaluation"], cfg["evaluation"]["protocol"]
    split = args.split or pc["split"]
    if split == "test" and not args.allow_test:
        print("Refusing to evaluate on the test split without --allow-test.", file=sys.stderr)
        return 2
    from src.recognition.crnn import latest_crnn_run, load_crnn
    crnn_dir = resolve_path(ev["crnn_run"]) if ev.get("crnn_run") else latest_crnn_run()
    if crnn_dir is None:
        print("No finished CRNN run (scripts/train_crnn.py) under models/checkpoints/.", file=sys.stderr)
        return 2
    embedder = get_writer_embedder("trained", cfg, run_dir=args.writer_id_run)
    if not embedder.usable:
        print(f"WARNING: writer-ID embedder {embedder.name} is NOT usable (test top-1 below threshold); "
              "its distances must not be quoted.", file=sys.stderr)
    scorer = P.Scorer(load_crnn(crnn_dir), embedder, HWDFeatures(), cfg["crnn"]["input_height"])

    dataset = load_iam(split, unit="lines", root=args.iam_root, splits_dir=args.splits_dir)
    by_writer: dict[str, list] = {}
    for s in dataset.samples:
        by_writer.setdefault(s.writer_id, []).append(s)
    ks = args.ks or pc["ks"]
    writers = P.choose_writers(dataset, pc["n_writers"], pc["seed"], max(ks) + pc["targets_per_writer"] // 2 + 1)
    subset = f"{len(writers)} of {len(dataset.writer_ids)} {split} writers (seed {pc['seed']})"
    pcfg = {"targets_per_writer": pc["targets_per_writer"], "min_reference_letters": ev["min_reference_letters"],
            "max_target_chars": ev["max_target_chars"]}
    git = git_info()
    utc = datetime.now(timezone.utc)
    results_dir = Path(args.results_dir) if args.results_dir else resolve_path(ev["results_dir"])
    table = results_dir / "protocol_table.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    cpu_reduced = True                                    # CUDA unavailable: recorded in ROADMAP_STATE.md

    for name in args.backends:
        backend = None if name == "real" else _make_backend(name, cfg)
        for K in ks:
            if backend is not None:                       # fresh backend per K: per-K coverage counters (Task 2.4)
                backend = _make_backend(name, cfg)
            rows, skipped, writer_errors = [], [], {}
            for w in writers:
                r = None
                snap = ((backend.coverage_seen.copy(), backend.coverage_missing.copy())
                        if hasattr(backend, "coverage_seen") else None)
                for attempt in range(2):              # a host-OOM (not a code error) gets one retry for the writer
                    if snap is not None:              # a failed attempt must not leave its counts in the coverage stat
                        backend.coverage_seen, backend.coverage_missing = snap[0].copy(), snap[1].copy()
                    try:
                        r = P.run_writer(scorer, backend, dataset, by_writer[w], K, pcfg, pc["seed"], w)
                        break
                    except Exception as e:           # noqa: BLE001 -- recorded, never silently dropped
                        logger.warning("writer %s failed (attempt %d): %r", w, attempt + 1, e)
                        writer_errors.setdefault(w, []).append(f"{type(e).__name__}: {e}")
                        if not P.is_out_of_memory(e):
                            break
                        gc.collect()
                if r is not None:
                    rows.append(r)
                else:
                    skipped.append(w)
                logger.info("%s K=%d writer %s -> %s", name, K, w, "ok" if r else "skipped")
            if not rows:
                print(f"{name} K={K}: no writer could be scored", file=sys.stderr)
                continue
            agg = P.aggregate(rows)
            run_id = f"{utc:%Y%m%dT%H%M%SZ}_{git['sha']}{'-dirty' if git['dirty'] else ''}_protocol_{name}_K{K}"
            out_dir = results_dir / run_id
            out_dir.mkdir(parents=True, exist_ok=False)
            record = {"run_id": run_id, "git": git, "split": split, "subset": subset, "cpu_reduced": cpu_reduced,
                      "backend": name, "K": K, "protocol": pc, "crnn_run": crnn_dir.name,
                      "writer_id": embedder.describe(), "aggregate": agg, "writers": rows, "skipped_writers": skipped,
                      "writer_errors": writer_errors,
                      "coverage": backend.coverage() if hasattr(backend, "coverage") else None,
                      "note": "real row = half of each writer's held-out real lines scored against the other half"}
            (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=float) + "\n",
                                                  encoding="utf-8")
            new = not table.exists()
            with table.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow({"run_id": run_id, "utc": utc.isoformat(), "git_sha": git["sha"], "git_dirty": git["dirty"],
                            "split": split, "subset": subset, "cpu_reduced": cpu_reduced, "backend": name, "K": K,
                            "n_writers": agg["n_writers"], "cer": agg["all_cer"], "wid_dist": agg["all_wid_dist"],
                            "hwd": agg["all_hwd"], "oov_n_writers": agg["oov_n_writers"], "oov_cer": agg["oov_cer"],
                            "oov_wid_dist": agg["oov_wid_dist"], "oov_hwd": agg["oov_hwd"],
                            "n_failed_generations": agg["n_failed_generations"], "crnn_run": crnn_dir.name,
                            "writer_id_run": embedder.run_id, "metrics_path": str(out_dir / "metrics.json")})
            print(f"{name:10s} K={K}: CER {agg['all_cer']:.3f} | writer-ID dist {agg['all_wid_dist']:.4f} | "
                  f"HWD {agg['all_hwd']:.3f} | OOV n={agg['oov_n_writers']} CER {agg['oov_cer']} "
                  f"({agg['n_writers']} writers, {subset}, CPU-reduced)")
    return 0
