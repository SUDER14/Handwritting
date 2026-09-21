"""Train the writer-ID embedding on IAM (train-split writers), select on val writers, report on TEST writers.

    python scripts/train_writer_id.py                       # full config
    python scripts/train_writer_id.py --epochs 2 --run-name smoke

Loss = cross-entropy over train-split writer IDs + `writer_id.loss.ntxent_weight` * multi-positive NT-Xent over the
embeddings of same-writer crops, so the space is usable as a distance and not merely separable.

Checkpoints go to models/checkpoints/<run_id>/ (never overwritten; see src/writer_id/runs.py). Model selection uses
retrieval on VAL writers; the TEST writers (unseen at training time) are evaluated once at the end. If test top-1 is
below `writer_id.eval.min_usable_top1` the script says so loudly and the embedding must not be used to quote a
style-fidelity number.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.data.iam import load_iam_splits
from src.data.leakage import assert_no_test_writers
from src.evaluation.harness import git_info
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger
from src.writer_id.data import LineStore, WriterBatchSampler
from src.writer_id.model import WriterIDNet, nt_xent_multi_positive
from src.writer_id.retrieval import embed_samples, retrieval_metrics
from src.writer_id.runs import best_checkpoint, load_encoder, new_run_dir, write_once

logger = get_logger(__name__)


def group_by_writer(samples) -> dict[str, list]:
    out: dict[str, list] = {}
    for s in samples:
        out.setdefault(s.writer_id, []).append(s)
    return out


def evaluate_split(encoder, store: LineStore, dataset, wid_cfg: dict) -> dict:
    samples = dataset.samples
    emb = embed_samples(encoder, store, samples, wid_cfg["eval"])
    return retrieval_metrics(emb, [s.writer_id for s in samples], [s.form_id for s in samples],
                             exclude_same_form=wid_cfg["eval"]["exclude_same_form"])


def banner(lines: list[str], char: str = "!") -> None:
    width = max(len(x) for x in lines) + 4
    print("\n" + char * width)
    for x in lines:
        print(f"{char} {x.ljust(width - 4)} {char}")
    print(char * width + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--iam-root", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--checkpoint-root", default=None, help="default: models/checkpoints")
    ap.add_argument("--skip-test", action="store_true", help="do not evaluate the test writers at the end")
    args = ap.parse_args()

    cfg = load_config()
    wid = copy.deepcopy(cfg["writer_id"])
    tr = wid["train"]
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        tr["steps_per_epoch"] = args.steps_per_epoch
    torch.set_num_threads(tr["num_threads"])
    torch.manual_seed(tr["seed"])
    rng = np.random.default_rng(tr["seed"])

    try:
        datasets = load_iam_splits(unit="lines", config=cfg, root=args.iam_root, splits_dir=args.splits_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    store = LineStore(resolve_path(wid["cache_dir"]), wid["input_height"], wid["binarize"])
    for name in ("train", "val"):
        store.ensure(datasets[name])
    test_ids = set(datasets["test"].writer_ids)
    assert_no_test_writers({x.writer_id for x in datasets["train"].samples}, test_ids, "training set")
    assert_no_test_writers({x.writer_id for x in datasets["val"].samples}, test_ids, "validation set")
    train_by_writer = group_by_writer(datasets["train"].samples)
    train_writers = sorted(train_by_writer)
    writer_to_idx = {w: i for i, w in enumerate(train_writers)}
    logger.info("train: %d writers, %d lines | val: %d writers, %d lines | test: %d writers, %d lines",
                len(train_writers), len(datasets["train"]), len(datasets["val"].writer_ids), len(datasets["val"]),
                len(datasets["test"].writer_ids), len(datasets["test"]))

    git = git_info()
    root = Path(args.checkpoint_root) if args.checkpoint_root else None
    run_id, run_dir = new_run_dir(git["sha"], git["dirty"], args.run_name, root)
    fingerprint = {n: hashlib.sha256(",".join(sorted(datasets[n].writer_ids)).encode()).hexdigest()[:16] for n in datasets}
    write_once(run_dir / "config.json", json.dumps({
        "kind": "writer_id", "run_id": run_id, "git": git, "writer_id": wid, "split_seed": cfg["data"]["iam"]["split_seed"],
        "split_writer_fingerprints": fingerprint, "num_train_writers": len(train_writers),
        "note": "no pretrained weights; ImageNet initialisation was NOT compared",
    }, indent=2) + "\n")
    logger.info("run %s -> %s", run_id, run_dir)

    m = wid["model"]
    net = WriterIDNet(len(train_writers), m["widths"], m["blocks"], m["stem_stride"])
    logger.info("model parameters: %.2fM", sum(p.numel() for p in net.parameters()) / 1e6)
    sampler = WriterBatchSampler(store, train_by_writer, writer_to_idx, tr["writers_per_batch"],
                                 tr["crops_per_writer"], wid["augment"], tr["seed"], forbidden_writers=test_ids)
    batch = min(tr["writers_per_batch"], len(train_writers)) * tr["crops_per_writer"]
    steps = tr["steps_per_epoch"] or math.ceil(len(datasets["train"]) / batch)
    total = steps * tr["epochs"]
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr["lr"], total_steps=total, pct_start=0.1)
    lo, hi = wid["window_width"]
    lam, temp = wid["loss"]["ntxent_weight"], wid["loss"]["temperature"]

    log_path = run_dir / "train_log.csv"
    log_f = open(log_path, "x", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_ce", "train_ntxent", "train_acc", "val_top1", "val_map", "val_chance", "seconds"])
    best = (-1.0, -1.0)
    best_epoch = None
    epochs_run = tr["epochs"]
    val_metrics = None
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        net.train()
        ce_sum = nt_sum = acc_sum = 0.0
        for _ in range(steps):
            width = int(rng.integers(lo // 8, hi // 8 + 1)) * 8
            x, y = sampler.sample(width)
            x = torch.from_numpy(x.astype(np.float32) / 255.0)[:, None]
            y = torch.from_numpy(y)
            emb, logits = net(x)
            ce = F.cross_entropy(logits, y)
            nt = nt_xent_multi_positive(emb, y, temp)
            loss = ce + lam * nt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ce_sum += ce.item(); nt_sum += nt.item(); acc_sum += (logits.argmax(1) == y).float().mean().item()
        val_top1 = val_map = val_chance = float("nan")
        if epoch % args.val_every == 0 or epoch == tr["epochs"]:
            val_metrics = evaluate_split(net.encoder, store, datasets["val"], wid)
            val_top1, val_map, val_chance = val_metrics["top1"], val_metrics["map"], val_metrics["chance_top1"]
            if (val_top1, val_map) > best:
                best, best_epoch = (val_top1, val_map), epoch
                torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "val": val_metrics,
                            "train_writers": train_writers}, run_dir / "best.pt")
        dt = time.time() - t0
        log.writerow([epoch, f"{ce_sum/steps:.4f}", f"{nt_sum/steps:.4f}", f"{acc_sum/steps:.4f}",
                      f"{val_top1:.4f}", f"{val_map:.4f}", f"{val_chance:.4f}", f"{dt:.1f}"])
        log_f.flush()
        torch.save({"model_state_dict": net.state_dict(), "epoch": epoch, "train_writers": train_writers}, run_dir / "last.pt")
        logger.info("epoch %d/%d | ce %.3f nt-xent %.3f train-acc %.3f | val top1 %.3f mAP %.3f (chance %.3f) | %.0fs",
                    epoch, tr["epochs"], ce_sum / steps, nt_sum / steps, acc_sum / steps, val_top1, val_map, val_chance, dt)
        if best_epoch is not None and epoch - best_epoch >= tr["early_stop_patience"]:
            logger.info("early stop: no val improvement for %d epochs (best epoch %d)", tr["early_stop_patience"], best_epoch)
            epochs_run = epoch
            break
    log_f.close()

    test_metrics = None
    if not args.skip_test:
        store.ensure(datasets["test"])
        encoder, _ = load_encoder(run_dir)
        test_metrics = evaluate_split(encoder, store, datasets["test"], wid)
        test_metrics.update({"split": "test", "checkpoint": best_checkpoint(run_dir).name, "eval_mode": wid["eval"]["mode"]})
        write_once(run_dir / "test_metrics.json", json.dumps(test_metrics, indent=2) + "\n")
    write_once(run_dir / "run.json", json.dumps({
        "run_id": run_id, "best_epoch": best_epoch, "best_val": {"top1": best[0], "map": best[1]},
        "epochs_cap": tr["epochs"], "epochs_run": epochs_run, "steps_per_epoch": steps, "batch_size": batch,
        "num_train_lines": len(datasets["train"]), "test": test_metrics,
    }, indent=2) + "\n")

    if test_metrics is not None:
        thr = wid["eval"]["min_usable_top1"]
        head = (f"TEST writers (unseen): top-1 retrieval {test_metrics['top1']:.3f}, mAP {test_metrics['map']:.3f} "
                f"(chance top-1 {test_metrics['chance_top1']:.3f}; {test_metrics['n_queries']} queries, "
                f"{test_metrics['n_writers']} writers, cross-form gallery={test_metrics['exclude_same_form']})")
        if test_metrics["top1"] < thr:
            banner([f"WRITER-ID EMBEDDING IS NOT USABLE: test top-1 {test_metrics['top1']:.3f} < {thr}", head,
                    "Do NOT quote any style-fidelity number computed with this embedding."])
        else:
            print(head + f"\nabove the {thr} usability threshold.")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
