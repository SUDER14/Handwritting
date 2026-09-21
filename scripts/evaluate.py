"""Evaluate the generators on writer-disjoint IAM data and persist the run.

Reports BOTH headline metrics, always together:
  * legibility     = CER of the CNN recognizer reading the generated text back (lower = better)
  * style fidelity = cosine distance between writer-ID embeddings of generated vs reference image (lower = better)
plus leave-one-out SSIM (secondary; the held-out glyph is excluded from the style profile).

Every run writes  results/<utc-timestamp>_<git-sha>/metrics.json  (run id, git SHA, full resolved config,
checkpoint paths + hashes, split name, per-sample and aggregate metrics) and appends one row to
results/index.csv so runs can be compared without parsing JSON.

Usage:
    python scripts/evaluate.py --split val                       # baseline generator, validation writers
    python scripts/evaluate.py --split val --backend neural
    python scripts/evaluate.py --split val --max-writers 10      # quick look
    python scripts/evaluate.py --split test --allow-test         # final numbers only; the test split is guarded
    python scripts/evaluate.py --smoke                           # synthetic font-rendered samples: pipeline smoke
                                                                 # test only, flagged is_smoke_test=true, NOT a result

Needs the IAM dataset under data/iam (see src/data/iam.py) and a trained CNN checkpoint
(scripts/train_cnn_recognizer.py). The writer-ID embedder is currently a STUB (see
src/evaluation/writer_id.py); runs record that, and style-fidelity numbers from a stub are not results.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_NAMES, load_iam
from src.evaluation import harness
from src.evaluation.writer_id import get_writer_embedder
from src.recognition.cnn_recognizer import CNNRecognizer
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["protocol", "legacy"], default="protocol",
                        help="protocol = ROADMAP 1.4 (K references -> the writer's other lines; CRNN CER + writer-ID + HWD); "
                             "legacy = the older single-reference CNN-glyph harness (kept for its tests)")
    parser.add_argument("--backends", nargs="+", default=["real", "baseline"],
                        help="protocol mode: any of real baseline glyph_vae (vatr in Stage 3)")
    parser.add_argument("--ks", nargs="+", type=int, default=None, help="protocol mode: override evaluation.protocol.ks")
    parser.add_argument("--split", choices=SPLIT_NAMES, default=None,
                        help="default: val (legacy) / evaluation.protocol.split (protocol)")
    parser.add_argument("--backend", choices=["baseline", "neural"], default="baseline")
    parser.add_argument("--max-writers", type=int, default=None, help="evaluate a deterministic subset of the split's writers")
    parser.add_argument("--smoke", action="store_true", help="run on data/samples/*.png (NOT an evaluation)")
    parser.add_argument("--allow-test", action="store_true", help="required to touch the test split")
    parser.add_argument("--iam-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--results-dir", default=None, help="default: evaluation.results_dir from config")
    parser.add_argument("--writer-embedder", choices=["trained", "stub"], default=None,
                        help="default: evaluation.writer_embedder from config. 'stub' is for smoke tests only.")
    parser.add_argument("--writer-id-run", default=None, help="writer-ID run dir (default: newest under models/checkpoints)")
    args = parser.parse_args()
    if args.mode == "protocol":
        from src.evaluation.protocol_cli import run_protocol_cli
        return run_protocol_cli(args)
    args.split = args.split or "val"

    cfg = load_config()
    resolved_config = copy.deepcopy(cfg)   # recorded verbatim in metrics.json
    ev = cfg["evaluation"]

    if not args.smoke and args.split == "test" and not args.allow_test:
        print("Refusing to evaluate on the test split without --allow-test. Use val while developing.", file=sys.stderr)
        return 2

    cnn_path = resolve_path(cfg["recognition"]["cnn_checkpoint_path"])
    vae_path = resolve_path(cfg["training"]["checkpoint_path"]) if args.backend == "neural" else None
    try:
        recognizer = CNNRecognizer(checkpoint_path=str(cnn_path))
    except FileNotFoundError as e:
        print(f"Cannot compute legibility (CER) without the CNN recognizer: {e}", file=sys.stderr)
        return 2
    if vae_path is not None and not vae_path.exists():
        print(f"--backend neural needs a VAE checkpoint at {vae_path}", file=sys.stderr)
        return 2
    try:
        embedder = get_writer_embedder(args.writer_embedder or ev["writer_embedder"], cfg, run_dir=args.writer_id_run)
    except (NotImplementedError, FileNotFoundError) as e:
        print(f"Cannot compute style fidelity: {e}", file=sys.stderr)
        return 2

    engine = None
    if args.backend == "neural":
        from src.style_encoder.inference import StyleEncoderInference
        engine = StyleEncoderInference(checkpoint_path=str(vae_path))
    ctx = harness.EvalContext(config=cfg, backend=args.backend, recognizer=recognizer, embedder=embedder,
                              neural_checkpoint=str(vae_path) if vae_path else None, neural_engine=engine)

    skipped_writers: list[dict] = []
    if args.smoke:
        split_name, unit = "smoke", "synthetic"
        print("*** SMOKE TEST on synthetic font-rendered samples. This is not an evaluation. ***")
        records = harness.run_smoke(ctx, resolve_path("data/samples"))
        split_files = None
    else:
        split_name, unit = args.split, "lines"
        try:
            dataset = load_iam(args.split, unit=unit, root=args.iam_root, splits_dir=args.splits_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        splits_dir = Path(args.splits_dir) if args.splits_dir else resolve_path(cfg["data"]["iam"]["splits_dir"])
        split_files = {n: harness.checkpoint_record(splits_dir / f"{n}.json") for n in SPLIT_NAMES}
        logger.info("Evaluating split=%s: %d writers, %d lines, backend=%s",
                    args.split, len(dataset.writer_ids), len(dataset), args.backend)
        records, skipped_writers = harness.run_iam_split(ctx, dataset, max_writers=args.max_writers)

    aggregate = harness.aggregate(records)   # raises rather than reporting one metric alone

    record = {
        "schema_version": harness.SCHEMA_VERSION,
        "git": harness.git_info(),
        "split": split_name,
        "unit": unit,
        "backend": args.backend,
        "is_smoke_test": bool(args.smoke),
        "writer_embedder": embedder.describe(),
        "cli_args": vars(args),
        "resolved_config": resolved_config,
        "checkpoints": {
            "cnn": harness.checkpoint_record(cnn_path),
            "vae": harness.checkpoint_record(vae_path),
        },
        "split_files": split_files,
        "skipped_writers": skipped_writers,
        "samples": records,
        "aggregate": aggregate,
    }
    results_dir = Path(args.results_dir) if args.results_dir else resolve_path(ev["results_dir"])
    metrics_path = harness.write_run(results_dir, record)

    a = aggregate
    print()
    print(f"split={split_name} backend={args.backend} samples={a['n_samples']} (failed {a['n_failed']}) writers={a['n_writers']}")
    print(f"  legibility     CER (micro / mean)          : {a['cer_micro']:.3f} / {a['cer_mean']:.3f}   (lower is better)")
    print(f"  style fidelity cosine dist vs reference    : {a['style_distance_reference_mean']:.4f}   (lower is better)")
    if a["style_distance_heldout_mean"] is not None:
        print(f"                 cosine dist vs held-out real: {a['style_distance_heldout_mean']:.4f}")
    if a["loo_ssim_mean"] is not None:
        print(f"  leave-one-out SSIM (secondary)             : {a['loo_ssim_mean']:.3f}")
    if embedder.is_stub:
        print(f"  NOTE: writer embedder is a STUB ({embedder.name}); style-fidelity numbers are placeholders.")
    elif not embedder.usable:
        tm = embedder.test_metrics
        top1 = "no test metrics recorded" if tm is None else f"test top-1 {tm['top1']:.3f} < {embedder.min_usable_top1}"
        print("  !" * 30)
        print(f"  WARNING: writer-ID embedder {embedder.name} is NOT USABLE ({top1}).")
        print("  Do NOT quote the style-fidelity numbers above.")
        print("  !" * 30)
    else:
        tm = embedder.test_metrics
        print(f"  writer embedder {embedder.name}: unseen-writer top-1 {tm['top1']:.3f}, mAP {tm['map']:.3f} "
              f"(chance {tm['chance_top1']:.3f})")
    if args.smoke:
        print("  NOTE: smoke test on synthetic samples; do not quote these numbers as results.")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
