# PROJECT_STATUS.md

Current state of the repository, written for someone (or an AI assistant) with no prior context. Rewritten 2026-10-25 for
the ROADMAP.md execution; the pre-ROADMAP audit of the original EMNIST/font prototype is in git history (`git log --
PROJECT_STATUS.md`) and is no longer accurate. Every number below is quoted from `ROADMAP_STATE.md`, `results/` or a log; the
source is named next to it.

> **GATE 3 IS OPEN (NOT PASSED).** The VATr reproduction gave HWD **0.6069** vs the published **0.828**
> (rel. diff 0.267; the gate needs 0.745-0.911). The VATr protocol rows (3.5) and samples (3.6) are **PENDING COLAB**.
> Work on Stages 4-6 continued past the open gate at the user's explicit, session-only request (logged in
> `ROADMAP_STATE.md`). The gate's thresholds were not changed and GATE 3 is not logged as passed anywhere.

## What this is

A study of few-shot, line-level handwriting synthesis on real IAM writers: three generators (font-glyph **baseline**, EMNIST
**glyph VAE**, **VATr**) compared under one protocol with frozen instruments for legibility (CRNN CER) and style fidelity
(from-scratch writer-ID net, and the official HWD). Host: Windows 10, 4-core CPU, 7.8 GB RAM, **no CUDA** -> all training
caps /4 ("CPU-reduced") and every protocol row is a **20-writer subset** (seed 1337) of the 161 test writers.

## Gate status (from ROADMAP_STATE.md)

| gate | state | key facts |
|---|---|---|
| 0 data | PASSED | join match rate 0.9024 (9360/10373), 0 conflicting form writers, 0 train/test writer overlap |
| 1 instruments | PASSED | writer-ID test top-1 0.7319 (>=0.60); HWD same<diff in 200/200 pairs (>=0.90); CRNN test CER 0.1023 (<=0.15); real row present; leakage assertion on |
| 2 baseline | PASSED | real/baseline/glyph_vae x K=1,5 rows present; shear test passes |
| **3 VATr** | **OPEN** | 3.2 reproduction 0.6069 vs 0.828 NOT PASSED; 3.5/3.6 VATr rows + samples pending Colab; 3.3 wrapper done; 3.4 fine-tuning skipped (no GPU) |
| 4 user sample | PASSED | word count within +-1 on 0.865 of 200 test lines (>=0.80); deskew synthetic test passes; 4.3 real photos DEFERRED [HUMAN] |
| 5 experiments | files present; VATr parts PENDING | 5.1 K-sweep (baseline), 5.2 per-writer, 5.3 metric agreement (rho 0.5767 overall, 0.2009 baseline, 0.1110 glyph_vae) done; 5.4 human ratings DEFERRED [HUMAN] |
| 6 product | done | app headless smoke tests pass, README + this file rewritten; full suite 185 passed, 1 skipped |

## Results table (results/protocol_table.csv)

20-writer seed-1337 test subset, pinned CPU-reduced instruments, lower is better:

| backend | K | CER | writer-ID dist | HWD |
|---|---|---|---|---|
| real | 1 / 5 | 0.1038 / 0.0934 | 0.0158 / 0.0215 | 0.771 / 0.898 |
| baseline | 1 / 5 | 0.8001 / 0.8352 | 0.6861 / 0.6765 | 3.443 / 3.463 |
| glyph_vae | 1 / 5 | 0.7954 / 0.8086 | 0.4528 / 0.4994 | 4.354 / 4.244 |
| VATr | 1 / 5 | PENDING COLAB | PENDING COLAB | PENDING COLAB |

## Code map

| path | role |
|---|---|
| `src/data/iam.py`, `scripts/join_teklia.py`, `make_vatr_split.py`, `materialize_iam.py` | IAM loader, Teklia-to-writer join, canonical VATr/HWT split, leakage assertion (`src/data/leakage.py`) |
| `src/writer_id/`, `scripts/train_writer_id.py` | from-scratch writer-ID ResNet (widths 32-256, 256-d embedding, CE + NT-Xent) = fidelity instrument |
| `src/recognition/crnn.py`, `scripts/train_crnn.py` | CRNN-CTC line recognizer = legibility instrument (frozen) |
| `src/evaluation/{protocol,protocol_cli,backends,hwd_metric,writer_id}.py`, `scripts/evaluate.py` | the 1.4 protocol, backends, official-HWD wrapper |
| `src/generator/baseline_generator.py`, `src/style_encoder/` | baseline and glyph-VAE generators |
| `src/generator/vatr_backend.py` | VATr behind the protocol interface; runtime shims only (`third_party/VATr` is unedited) |
| `src/segmentation/words.py`, `src/preprocessing/` | Stage 4: line -> words -> K references; deskew |
| `src/analysis/writer_features.py`, `scripts/analyze_per_writer.py`, `plot_k_sweep.py`, `metric_agreement.py`, `analyze_metric_agreement.py`, `make_human_check.py`, `merge_vatr_rows.py` | Stage 5 |
| `src/app_logic.py`, `app/app.py` | Stage 6 Streamlit app (`app/legacy_app.py` = the old alphabet UI) |
| `scripts/vatr_reproduce.py`, `vatr_hwd_audit.py`, `vatr_smoke.py`, `notebooks/colab_train.ipynb` | VATr on Colab / the GATE 3 audit |
| `tests/` | full suite: 185 passed, 1 skipped (accelerate not installed) |

## Known issues and open items

1. **GATE 3 / task 3.2.** See `docs/gate3_audit.md`. Hypotheses (untested, need the Colab images): VATr writes white-padded
   images into `Real_test`/`Fake_test` whose padding depends on the batch size (Colab used 32); the 2023 and 2025 HWD code
   pool differently. Next: run `scripts/vatr_hwd_audit.py` on Colab and send `results/vatr_hwd_audit.json`.
2. **VATr rows** (3.5) and **samples** (3.6): Colab cells 10b/10c; merge with `scripts/merge_vatr_rows.py` (refuses unpinned instruments).
   Note: `G:\My Drive\handwriting\instruments\crnn_..._cpu_r1/` had no `best.pt` on 2026-10-20 (ROADMAP_STATE), so cell 10b cannot run until it is uploaded.
3. **4.3 real photos**: needs `data/user_photos/` ([HUMAN]). Not started.
4. **5.4 human ratings**: `results/human_check/` built, waiting for the user's ratings ([HUMAN]).
5. **forms.txt** is in public git history (commit `5150ef8`); history not rewritten by the user's choice.
6. The synthetic backends are near-illegible (CER about 0.8); the writer-ID net and HWD disagree about which is closer to the writer (`docs/metric_agreement.md`).
7. Host memory is tight (about 0.1-1.3 GB free): run one heavy process at a time, detached, logging to `logs/`; two earlier runs were reaped by the harness.

## How to resume

Read `ROADMAP_STATE.md` bottom-up (RESUME HERE), then `ROADMAP.md`. Never change a gate threshold; every number written to a
doc must come from a file in `results/` or a log.
