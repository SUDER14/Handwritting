# Handwriting style synthesis: line-level generation, scored on unseen IAM writers

Given a line of someone's handwriting, generate arbitrary text in that handwriting, and measure two things on real writers
the system never trained on: **legibility** (can a frozen recognizer read it?) and **style fidelity** (does it look like
that writer?). Three generators are compared under one protocol: a non-learned font-glyph **baseline**, an EMNIST **glyph
VAE** (ablation) and **VATr** (Pippi et al.), the line-level few-shot model, run with its released checkpoint.

> **Status (read this first).** Everything below the line "Results" was produced on a CPU-only 4-core host, so every
> trained component is **CPU-reduced** (training caps divided by 4) and every table row is a **20-writer subset** of the 161
> test writers. **GATE 3 is OPEN**: the VATr reproduction gave HWD **0.6069 against the published 0.828** (outside the
> 10% band 0.745-0.911), and the VATr protocol rows are **PENDING COLAB**. No VATr number is in this README that did not
> come from a file. See `docs/gate3_audit.md` and `FINAL_REPORT.md`.

## Data provenance

* **IAM Handwriting Database** (Marti & Bunke, *The IAM-database: an English sentence database for offline handwriting
  recognition*, IJDAR 5(1), 2002). Licensed; **never committed** (`data/iam/*` is gitignored).
* **Lines and text**: Hugging Face `Teklia/IAM-line` (image + text, no writer id) and `xReniar/IAM-Dataset` (line id + text
  per form). `scripts/join_teklia.py` joins them to writers through IAM's official `forms.txt` (form -> writer): a Teklia
  row is kept only when its text matches exactly one line id inside the writer group its split allows. Match rate 0.9024
  (9360 of 10373 rows), 1013 rows excluded and logged, zero forms with conflicting writers.
* **`forms.txt`** is IAM's own file (supplied by hand; licensed). It was **committed by mistake in commit `5150ef8`** and is
  therefore in the **public git history** of this repository (it is untracked and gitignored since `5457f3b`, but the
  history was deliberately not rewritten). Treat that file as published.
* **Canonical split**: the IAM writer split of HWT/VATr (283 train / 56 val / 161 test writers; `data/splits/vatr/*.json`,
  made by `scripts/make_vatr_split.py`). Test writers are never seen by any trained component (asserted in the data loader).
* **HWD** (Handwriting Distance): Pippi et al., BMVC 2023, official repo `aimagelab/HWD` at `eabb5b5` (pretrained VGG16
  backbone, downloaded by the official code, not stored here).
* **VATr**: Pippi et al., *Handwritten Text Generation from Visual Archetypes*, CVPR 2023, official repo `aimagelab/VATr`
  at `183501e` with the released checkpoint. Both repos are cloned into `third_party/` (gitignored) and are **not edited**;
  all adaptations are runtime shims (`src/generator/vatr_backend.py`).

## Setup

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m pytest tests -q          # full suite
```

IAM goes under `data/iam/` (layout in `docs/iam_setup.md`); put the licensed `forms.txt` at the repo root. Use
`venv\Scripts\python` for everything (system Python lacks OpenCV). No GPU is needed for the baseline/glyph_vae/measurement
stages; VATr generation needs CUDA (`notebooks/colab_train.ipynb`).

## Commands per stage

| Stage | Command |
|---|---|
| 0 data | `python scripts/join_teklia.py` then `python scripts/make_vatr_split.py`, `python scripts/materialize_iam.py`, `python scripts/check_split.py` |
| 1.1 writer-ID net | `python scripts/train_writer_id.py` (config cap 10 epochs = 40/4) |
| 1.2 HWD sanity | `python scripts/hwd_sanity.py` -> `results/hwd_sanity.json` |
| 1.3 recognizer | `python scripts/train_crnn.py` (config cap 15 epochs = 60/4) |
| 1.4 / 2 / 3.5 protocol | `python scripts/evaluate.py --mode protocol --backends real baseline glyph_vae --allow-test` (add `vatr` on Colab) |
| 3.2 VATr reproduction | Colab: `python scripts/vatr_reproduce.py --device cuda --batch_size 32` (`--skip-generation` reuses images) |
| 3.2 audit | Colab: `python scripts/vatr_hwd_audit.py --device cuda` (see `docs/gate3_audit.md`) |
| 3.6 samples | `python scripts/make_samples.py --backends real baseline glyph_vae --allow-test` |
| 4 user sample -> references | `python scripts/stage4_measure.py words --allow-test`, `... deskew --allow-test` |
| 5.1 K-sweep | `python scripts/evaluate.py --mode protocol --backends baseline --ks 3 10 15 --allow-test`, then `python scripts/plot_k_sweep.py` |
| 5.2 per-writer | `python scripts/analyze_per_writer.py` |
| 5.3 metric agreement | `python scripts/metric_agreement.py` then `python scripts/analyze_metric_agreement.py` |
| 5.4 human check | `python scripts/make_human_check.py` (you rate `results/human_check/human_check.csv`) |
| merge Colab VATr rows | `python scripts/merge_vatr_rows.py --colab-table <protocol_table.csv> --colab-results <dir> [--allow-sweep]` |
| 6.1 app | `venv\Scripts\streamlit run app/app.py` |

## Results

All rows: IAM **test** split, the seeded **20-writer subset** (seed 1337, 20 of 161 writers), 10 target lines per writer,
instruments pinned to the GATE-1 runs (`writerid_20260921T180911Z_ed897d02_cpu_r1`, `crnn_20260922T003942Z_d1f87b49_cpu_r1`),
all instruments **CPU-reduced** (writer-ID 10 epochs, CRNN 15 epochs, i.e. the ROADMAP caps /4). Lower is better for all
three columns. Source: `results/protocol_table.csv` (each row has its `metrics.json` under `results/<run_id>/`).

| backend | K | CER (legibility) | writer-ID dist | HWD | how produced |
|---|---|---|---|---|---|
| real (upper bound) | 1 | 0.1038 | 0.0158 | 0.771 | half of the held-out real lines vs the other half; 20-writer subset, CPU-reduced instruments |
| real (upper bound) | 5 | 0.0934 | 0.0215 | 0.898 | same |
| baseline (font glyphs) | 1 | 0.8001 | 0.6861 | 3.443 | non-learned; 20-writer subset, CPU-reduced instruments |
| baseline (font glyphs) | 5 | 0.8352 | 0.6765 | 3.463 | same |
| glyph_vae (EMNIST VAE, ablation) | 1 | 0.7954 | 0.4528 | 4.354 | VAE trained on 6000 EMNIST images x 6 epochs (CPU-reduced); 20-writer subset |
| glyph_vae (EMNIST VAE, ablation) | 5 | 0.8086 | 0.4994 | 4.244 | same |
| **VATr** | 1 | **PENDING COLAB** | **PENDING COLAB** | **PENDING COLAB** | Colab GPU, 20-writer subset, pinned (CPU-reduced) instruments |
| **VATr** | 5 | **PENDING COLAB** | **PENDING COLAB** | **PENDING COLAB** | same |

Notes that matter when reading the table:

* The two synthetic backends are close to illegible to the frozen recognizer (CER about 0.8); the real row is the floor
  of what these instruments can measure (CER about 0.10).
* **glyph_vae beats baseline on writer-ID distance but loses on HWD.** The two style metrics disagree; the analysis of why
  is in `docs/metric_agreement.md`.
* K=1 and K=5 real rows differ because the seeded reference/target draw depends on K.

### VATr reproduction (GATE 3, open)

| | value |
|---|---|
| published HWD, VATr on IAM test (Pippi et al. 2023, Table 2) | 0.828 |
| reproduced here (Colab, 25766 image pairs, 161 writers, pinned HWD, `--batch_size 32`) | **0.6069** (reported by the user from `results/vatr_reproduction.json`; the file is not on the CPU host) |
| relative difference | 0.267 (gate: within 10%, i.e. 0.745-0.911) -> **NOT PASSED** |

The result is *better* than published, which points at a protocol difference rather than a generation bug. The leading,
untested hypothesis is the white padding VATr writes into both image folders (batch-dependent). `docs/gate3_audit.md`
lists the paper-vs-ours protocol diff and the exact Colab cells that test it. The gate and its thresholds are unchanged.

### K-sweep, per-writer analysis, metric agreement

* K-sweep (baseline; VATr pending): `results/figures/k_sweep.png`, `results/figures/k_sweep_baseline.csv`.
* Per-writer correlations with slant / stroke width / cursiveness: `results/figures/per_writer_correlations.csv`,
  `results/figures/per_writer_{baseline,glyph_vae}.png`.
* Writer-ID vs HWD agreement and the explanation tests: `docs/metric_agreement.md`, `results/metric_agreement/`,
  `results/figures/metric_{scatter,disagreement}.png`.

## App

`venv\Scripts\streamlit run app/app.py`: choose an IAM demo writer or upload a line (plus its text), pick
real / baseline / glyph_vae, type text, and see the output with its CER, writer-ID distance and HWD. VATr is a switch that
stays disabled, with the reason shown, unless the released checkpoint and a CUDA GPU are present. The first score loads
the frozen instruments (including the 500 MB HWD backbone) and takes a while on CPU. The earlier alphabet-building UI is in
`app/legacy_app.py`.

## Deviations and limits (full list in `FINAL_REPORT.md`)

* No CUDA on the development host: all training caps /4, all trained results CPU-reduced, evaluation on a 20-writer subset.
* The writer-ID net is a ResNet-18 block layout with widths 32/64/128/256 (2.87M parameters), not the standard 64-512.
* Setting up the first VATr smoke test downloaded torchvision's ImageNet `resnet18` (45 MB, not on the allow-list) through
  VATr's constructor; its weights are overwritten by the checkpoint, no result depends on it, and a shim now prevents it.
* `forms.txt` is in public git history (above).
* Deferred: real-photo test of the user-sample pipeline (4.3, needs `data/user_photos/`), human ratings (5.4), VATr rows (Colab).

## Repository map

```
app/            Streamlit app (app.py) + the old alphabet UI (legacy_app.py)
config/         config.yaml (pinned instruments, protocol, paths)
src/            data/ evaluation/ generator/ recognition/ segmentation/ preprocessing/ writer_id/ analysis/ ...
scripts/        one script per stage (table above)
results/        protocol_table.csv, per-run metrics.json, figures/, metric_agreement/, human_check/
docs/           architecture, methodology, iam_setup, gate3_audit, metric_agreement, ...
notebooks/      colab_train.ipynb (GPU runs: training, VATr reproduction, VATr protocol rows)
ROADMAP.md / ROADMAP_STATE.md / PROJECT_STATUS.md / FINAL_REPORT.md
```
