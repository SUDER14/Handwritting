"""Train the CRNN-CTC legibility instrument (Task 1.3) on TRAIN-split writers; select on VAL CER; report on TEST lines.

    python scripts/train_crnn.py [--run-name cpu_r1] [--epochs N]

Checkpoints: models/checkpoints/crnn_<utc>_<sha>_<name>/{best.pt,last.pt,config.json,train_log.csv,run.json,test_metrics.json}
(never overwriting another run). Greedy CTC decoding; CER/WER = corpus edit distance / reference length.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from src.data.iam import load_iam_splits
from src.data.leakage import assert_no_test_writers
from src.evaluation.harness import git_info
from src.recognition.crnn import CRNN, Recognizer, build_charset, cer_wer, encode, pad_batch
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger
from src.writer_id.data import LineStore
from src.writer_id.runs import new_run_dir, write_once

logger = get_logger(__name__)


def load_crops(store: LineStore, dataset, max_width: int):
    crops, texts = [], []
    for s in dataset.samples:
        c = np.asarray(store.load(s.sample_id))
        if c.shape[1] <= max_width and s.transcription.strip():
            crops.append(c)
            texts.append(s.transcription)
    return crops, texts


def evaluate(model, charset, crops, texts):
    hyps = Recognizer(model, charset).transcribe(crops)
    cer, wer = cer_wer(texts, hyps)
    return {"cer": cer, "wer": wer, "n_lines": len(texts), "examples": list(zip(texts[:3], hyps[:3]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--checkpoint-root", default=None)
    args = ap.parse_args()

    cfg = load_config()
    c = copy.deepcopy(cfg["crnn"])
    if args.epochs is not None:
        c["epochs"] = args.epochs
    torch.set_num_threads(c["num_threads"])
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])

    datasets = load_iam_splits(unit="lines", config=cfg)
    test_ids = set(datasets["test"].writer_ids)
    for name in ("train", "val"):
        assert_no_test_writers({s.writer_id for s in datasets[name].samples}, test_ids, f"{name} set")
    wid = cfg["writer_id"]
    store = LineStore(resolve_path(wid["cache_dir"]), c["input_height"], wid["binarize"])
    for name in datasets:
        store.ensure(datasets[name])
    tr_x, tr_t = load_crops(store, datasets["train"], c["max_line_width"])
    va_x, va_t = load_crops(store, datasets["val"], c["max_line_width"])
    charset = build_charset(tr_t)
    char_to_id = {ch: i + 1 for i, ch in enumerate(charset)}
    logger.info("train %d lines, val %d lines, charset %d", len(tr_x), len(va_x), len(charset))

    git = git_info()
    root = Path(args.checkpoint_root) if args.checkpoint_root else None
    run_id, run_dir = new_run_dir(git["sha"], git["dirty"], args.run_name, root, prefix="crnn")
    write_once(run_dir / "config.json", json.dumps({"kind": "crnn", "run_id": run_id, "git": git, "crnn": c,
                                                    "charset": charset, "cuda": torch.cuda.is_available()}, indent=2) + "\n")
    model = CRNN(len(charset), c["hidden"])
    logger.info("run %s | params %.2fM", run_id, sum(p.numel() for p in model.parameters()) / 1e6)
    steps = math.ceil(len(tr_x) / c["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps * c["epochs"], pct_start=0.15)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    log_f = open(run_dir / "train_log.csv", "x", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "val_cer", "val_wer", "seconds"])
    best_cer, best_epoch, epochs_run = float("inf"), None, c["epochs"]
    for epoch in range(1, c["epochs"] + 1):
        t0 = time.time()
        model.train()
        # bucket by width: shuffle, chunk into groups, sort inside each group so padding stays small
        order = rng.permutation(len(tr_x))
        group = c["batch_size"] * 8
        batches = []
        for g in range(0, len(order), group):
            chunk = sorted(order[g:g + group], key=lambda i: tr_x[i].shape[1])
            batches += [chunk[b:b + c["batch_size"]] for b in range(0, len(chunk), c["batch_size"])]
        rng.shuffle(batches)
        loss_sum = 0.0
        for idx in batches:
            crops = []
            for i in idx:
                cr = tr_x[i]
                f = 1.0 + rng.uniform(-c["width_jitter"], c["width_jitter"])
                crops.append(cv2.resize(cr, (max(8, int(cr.shape[1] * f)), cr.shape[0]), interpolation=cv2.INTER_AREA))
            x, in_len = pad_batch(crops)
            tgt = [encode(tr_t[i], char_to_id) for i in idx]
            logp = model(x)
            loss = ctc(logp, torch.tensor([k for t in tgt for k in t]), in_len, torch.tensor([len(t) for t in tgt]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
        val = evaluate(model, charset, va_x, va_t)
        dt = time.time() - t0
        log.writerow([epoch, f"{loss_sum / len(batches):.4f}", f"{val['cer']:.4f}", f"{val['wer']:.4f}", f"{dt:.0f}"])
        log_f.flush()
        logger.info("epoch %d/%d | ctc %.3f | val CER %.3f WER %.3f | %.0fs", epoch, c["epochs"], loss_sum / len(batches),
                    val["cer"], val["wer"], dt)
        torch.save({"model_state_dict": model.state_dict(), "charset": charset, "epoch": epoch}, run_dir / "last.pt")
        if val["cer"] < best_cer:
            best_cer, best_epoch = val["cer"], epoch
            torch.save({"model_state_dict": model.state_dict(), "charset": charset, "epoch": epoch, "val": val},
                       run_dir / "best.pt")
        elif epoch - best_epoch >= c["early_stop_patience"]:
            logger.info("early stop (best epoch %d, val CER %.3f)", best_epoch, best_cer)
            epochs_run = epoch
            break
    log_f.close()

    te_x, te_t = load_crops(store, datasets["test"], c["max_line_width"])
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test = evaluate(model, charset, te_x, te_t)
    test.update({"split": "test", "checkpoint": "best.pt", "best_epoch": best_epoch,
                 "skipped_wider_than": c["max_line_width"], "n_test_lines_total": len(datasets["test"])})
    write_once(run_dir / "test_metrics.json", json.dumps(test, indent=2, ensure_ascii=False) + "\n")
    write_once(run_dir / "run.json", json.dumps({"run_id": run_id, "best_epoch": best_epoch, "best_val_cer": best_cer,
                                                 "epochs_cap": c["epochs"], "epochs_run": epochs_run, "test": test},
                                                indent=2, ensure_ascii=False) + "\n")
    print(f"TEST (real lines, unseen writers): CER {test['cer']:.4f} WER {test['wer']:.4f} on {test['n_lines']} lines")
    print(f"run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
