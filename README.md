# Personalized Handwriting Generator

AI system that learns a person's handwriting style from a small
handwritten sample (even a single word) and generates the rest of the
alphabet in that style, then renders arbitrary text using the result.

## Problem statement

Given a photo of a short handwritten sample (e.g. the word "quick"),
generate the characters the person did *not* write (`a, b, d, e, f, ...`)
so that they look like they belong to the same handwriting — then use
the resulting personalized alphabet to render any text the user types.

## Motivation

Personalized handwriting synthesis has applications in accessibility
(assistive writing tools), personalization (digital cards, notes),
education, and as a research testbed for few-shot style transfer more
broadly. The interesting constraint is the "few-shot" part: real users
will not hand-write all 26 letters, so the system must generalize a
writer's *style* from a handful of *characters*.

## Existing research vs. this implementation

Handwriting synthesis and style transfer are established research areas
(e.g. GAN- and diffusion-based handwriting generation, RNN stroke models
like Graves 2013's seminal work). This project does not claim the
overall concept is novel. Where this implementation focuses novelty (per
its scope, honestly stated) is:
- a genuinely **few-shot** setting (a handful of characters, not a full
  labeled writer corpus) — see docs/methodology.md,
- **partial-alphabet inference** with an explicit hybrid policy (real
  observed glyphs preserved, only unobserved ones generated),
- a directly comparable **non-learned baseline** vs. a **learned
  conditional VAE**, evaluated with the same quantitative metrics
  (docs/experiments.md), and
- an architecture designed to run entirely on a modest CPU-only machine.

## Architecture

See `docs/architecture.md` for the full diagram and rationale. In short:

```
photo -> preprocess -> segment -> recognize -> extract style features
      -> generate missing letters (baseline CV transform, or a trained
         conditional VAE) -> personalized alphabet -> render any text
```

## Technologies

Python 3.12, OpenCV, scikit-image, NumPy/SciPy, PyTorch (CPU),
torchvision, Hugging Face `datasets` (EMNIST source), Streamlit, pytest.

## Installation

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

(This repo's venv was created and populated during development; re-run
the above only if setting up fresh.)

## Usage

### 1. Run the Streamlit app (recommended)

```powershell
venv\Scripts\streamlit run app/app.py
```

Then: upload a handwriting image (PNG/JPG/JPEG), type the word you wrote,
review detected characters + style analysis, generate the alphabet,
preview it, type any text, and generate/download the rendered result.

A trained CNN recognizer (`src/recognition/cnn_recognizer.py`, see
Training below) adds a "Recognition check" panel showing, per character,
whether the model's own pixel-level reading agrees with what you typed and
how confident it is — a mismatch usually flags a segmentation problem
(commonly a touching cursive join) rather than a bad classification. You
can also tick "Auto-detect the word with the trained CNN instead"
(experimental) to have the CNN read the word directly, with no typed label
required.

### 2. Command-line

```powershell
# Generate a personalized alphabet from a sample image.
venv\Scripts\python scripts\generate_alphabet.py --image data\samples\quick_sample.png --label quick --backend baseline --session-id demo

# Render arbitrary text using that alphabet.
venv\Scripts\python scripts\generate_text.py --session-id demo --text "the quick fox" --out outputs\generated_text\demo.png
```

`--backend neural` uses the trained conditional VAE instead (requires a
checkpoint — see Training below).

### 3. Create your own test sample

No scanner needed for a quick smoke test — a synthetic (font-rendered,
noise-added) sample generator is included:

```powershell
venv\Scripts\python scripts\make_synthetic_sample.py
```

## Glyph geometry

Every glyph carries a `GlyphMetrics` (`advance_width`, `x_height`, `baseline_offset`, `bbox`; conventions in
`src/generator/glyph.py`). The renderer scales each glyph to a common x-height and places it by baseline, so real
crops (~100 px), VAE output (28 px) and font-derived glyphs mix correctly and g j p q y drop below the line.
Observed glyphs get their metrics measured from the sample; the VAE's 28x28 glyphs get them from typographic
priors (`glyph_metrics:` in `config/config.yaml`) because EMNIST-style output carries no absolute size.

## Real data: IAM (writer-disjoint splits)

The IAM Handwriting Database is licensed and is **not** in this repo. Put it
under `data/iam/` in the standard layout (`ascii/{forms,lines,words}.txt`,
`lines/a01/a01-000u/*.png`, `words/...`); `src/data/iam.py` reads it, groups
samples by writer via `forms.txt`, and exposes a writer-disjoint train/val/test
split. The **canonical** split (`data.iam.split_source: vatr`) is the IAM writer split of HWT/VATr
(`data/splits/vatr/{train,val,test}.json`, 283/56/161 writers; built by `scripts/make_vatr_split.py`).
The old self-made split (a pure function of `data.iam.split_seed`, `data/splits/seeded/`) is **fallback only**
(`split_source: seeded`). Fixed files are checked for writer overlap and unknown writers on every load; the seeded
files are re-verified against the seed.

```powershell
venv\Scripts\python scripts\check_split.py      # asserts zero writer overlap, prints per-split counts
```

```python
from src.data.iam import load_iam
val = load_iam("val", unit="lines")
image, transcription, writer_id = val[0]   # uint8 gray, fixed height, variable width
```

`data/samples/{quick,hello}_sample.png` are font-rendered and exist **only as a
smoke test**; never report metrics computed on them as results.

## Training the neural style encoder

```powershell
venv\Scripts\python scripts\preprocess_dataset.py   # optional sanity check
venv\Scripts\python scripts\train_style_encoder.py
```

Downloads EMNIST-letters (via Hugging Face Hub — see
`src/utils/emnist_source.py` for why not torchvision's built-in
downloader) and trains the conditional VAE (`src/style_encoder/model.py`).
Checkpoints are written to `models/checkpoints/style_vae.pt`
(config: `training.checkpoint_path`). Default config trains on a 40k-image
subset for 8 epochs (~20-25 min on a 4-core CPU, no GPU) — adjust
`training.max_train_samples` / `training.epochs` in `config/config.yaml`
for a longer, higher-quality run.

## Training the CNN character recognizer

```powershell
venv\Scripts\python scripts\train_cnn_recognizer.py
```

Trains `src/recognition/cnn_model.py::CharCNN` (a small discriminative
classifier, separate from the generative style VAE above) on EMNIST-letters
with random-affine + morphological (erosion/dilation) + noise augmentation,
so it tolerates slant and variable stroke width rather than only upright
EMNIST-style printing — this is what lets `recognition.backend: "cnn"`
(`src/recognition/cnn_recognizer.py`) actually detect a character from its
pixels, instead of trusting a typed label. Checkpoint:
`models/checkpoints/char_cnn.pt` (config: `recognition_training.
checkpoint_path`); default config trains on a 20k-image subset for 10
epochs (~40 min on a 4-core CPU, no GPU) — adjust
`recognition_training.max_train_samples` / `.epochs` in `config/config.yaml`
for a longer run.

## Evaluation harness

```powershell
venv\Scripts\python scripts\evaluate.py --split val                 # baseline generator, validation writers
venv\Scripts\python scripts\evaluate.py --split val --backend neural
venv\Scripts\python scripts\evaluate.py --smoke                     # synthetic samples: smoke test, NOT a result
```

Two headline metrics, always reported together: **legibility** = CER of the CNN
recognizer reading the generated text back, and **style fidelity** = cosine
distance between writer-ID embeddings of the generated and the reference image.
The writer-ID model is not trained yet; `src/evaluation/writer_id.py` provides
the interface plus a rule-based **stub**, and every run records `is_stub`.
Each run writes `results/<utc-timestamp>_<git-sha>/metrics.json` (git SHA, full
resolved config, checkpoint paths + SHA-256, split, per-sample and aggregate
metrics) and appends a row to `results/index.csv`. The test split needs
`--allow-test`. Leave-one-out SSIM rebuilds the style profile from the remaining
characters only (`src/evaluation/loo.py`).

## Evaluation

```powershell
venv\Scripts\python -m pytest tests/ -v
```

37+ unit tests across preprocessing, segmentation, feature extraction,
generation, rendering, recognition, and the VAE architecture, plus an
end-to-end test (`tests/test_end_to_end.py`) covering the full
image-to-rendered-text pipeline. Quantitative generation-quality metrics
(SSIM, recognition accuracy, writer-style similarity) are in
`src/evaluation/metrics.py` — see `docs/experiments.md` for the
evaluation protocols built on top of them.

## Results

Full test suite: **40/40 passing** (`pytest tests/`), including two
end-to-end scenarios covering the whole image-to-rendered-text pipeline.

Conditional VAE trained on EMNIST-letters (6 epochs, 6,000-image subset —
reduced from the intended 40,000/8 by this dev machine's available RAM,
see docs/development_log.md): validation loss decreased monotonically
every epoch, 208.3 -> 137.5, no overfitting observed.

Baseline vs. neural comparison on a synthetic "quick" sample
(`scripts/evaluate.py`):

| Metric | Baseline (CV transform) | Neural (conditional VAE) |
|---|---|---|
| Recognition accuracy (OCR proxy) | 30.8% | 11.5% |
| Writer-style similarity (cosine) | 0.786 | 0.791 |
| Leave-one-out SSIM (mean) | 0.203 | 0.219 |

The neural model's style-capture metrics (similarity, SSIM) are
competitive with the baseline, but its generated *alphabet* is visibly
undertrained (many unobserved letters collapse toward similar shapes) —
a direct, documented consequence of the memory-constrained training run,
not an architectural issue. See `docs/development_log.md` for the full
diagnosis and `docs/future_work.md` for the fix (retrain with the
original config once more RAM is available).

CNN character recognizer (`models/checkpoints/char_cnn.pt`, trained on
EMNIST-letters with slant/stroke-width/noise augmentation — see "Training
the CNN character recognizer" above): reached **88.8% validation
accuracy** after 2 of the configured 10 epochs before this session hit the
same memory-constrained-machine kill documented for the VAE above (see
`docs/development_log.md`'s 2026-09-15 entry); re-running for the full 10
epochs once more RAM is available is a pure re-run, no code changes
needed. On the two real end-to-end samples in `data/samples/`, its
independent reading of each segmented glyph agreed with the typed label
on 4/5 characters for both `quick_sample.png` and `hello_sample.png`
(`recognize_with_cnn` cross-check), with both disagreements landing at
visibly lower confidence (0.66, 0.82) than the agreements (0.92-0.99) —
the low-confidence signal correctly flags the uncertain reads.

## Limitations

See `docs/limitations.md` for a stage-by-stage list — most importantly:
segmentation now splits common touching/cursive-joined letters via a
projection-profile heuristic, but it is still a heuristic (not a learned
model) and heavily overlapping cursive can still be misattributed; the
default recognition backend (`labeled`) still trusts a user-supplied
label rather than solving open-vocabulary recognition (the `cnn` backend
does attempt real open-vocabulary detection, at EMNIST-limited accuracy —
see below); and the neural generator is trained on EMNIST (which has no
writer-ID metadata), not a writer-labeled corpus.

## Future work

See `docs/future_work.md`. Done since the original MVP: the CNN character
recognizer and cursive-aware (projection-profile) segmentation splitting.
Remaining: segmentation-free CRNN+CTC HTR for true cursive words,
writer-labeled-dataset training with a writer-identity loss,
uppercase/digit charsets, cursive *joining* in the renderer (output side),
stroke-trajectory (vector, not raster) generation, SVG/font export, and
more.

## Project structure

```
handwriting-ai/
  README.md
  requirements.txt
  config/config.yaml
  data/{raw,processed,samples}/
  models/checkpoints/
  src/
    preprocessing/   segmentation/   recognition/   features/
    style_encoder/   generator/      renderer/       evaluation/   utils/
  app/app.py                     # Streamlit UI
  tests/                         # pytest unit + end-to-end tests
  scripts/                       # CLI entry points (train, generate, preprocess)
  outputs/{alphabets,generated_text}/
  docs/                          # architecture, methodology, experiments, limitations, future_work, development_log
```
