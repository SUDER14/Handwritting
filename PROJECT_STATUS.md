# PROJECT_STATUS.md

Audit of the repository at `handwriting matching/`, written for an AI assistant with **zero prior context**.
Audit date: 2026-09-20. Sources: reading every file under `src/`, `scripts/`, `app/`, `tests/`, `config/`;
running `pytest tests/` (47 passed) and `scripts/evaluate.py` on both sample images; loading both
checkpoints. **Not run by the auditor:** the Streamlit app, `train_style_encoder.py`,
`train_cnn_recognizer.py`, `preprocess_dataset.py`. Where something is absent this document says "not present".

> **Warning: `AGENTS.md` at the repo root describes a different project** (time-dependent emergency-vehicle
> routing: Dijkstra/A*, `perp/`, `make test`, `run_experiments.py`, two 3D HTML pages). None of those files
> exist here. Do not follow it. `README.md` and `docs/` describe the real project but are partly stale (see §7).
>
> Git: branch `master`, **zero commits**; every file is untracked. Platform: Windows 10, no GPU.

---

> ## Update after the data / evaluation / geometry work (3 commits after the baseline)
>
> This report describes the repo **as audited**. These parts are now out of date; `git log -4` has the details.
>
> - **AGENTS.md is deleted** (it belonged to a different project).
> - **§2/§4:** new `src/data/iam.py` (IAM loader, writer-disjoint seed-deterministic split, `data/splits/*.json`),
>   `scripts/check_split.py`, `src/evaluation/{harness,loo,text_metrics,writer_id}.py`, `src/generator/glyph.py`
>   (`GlyphMetrics`, `GlyphBitmap`), `src/features/line_metrics.py`. IAM itself is **not in the repo**; nothing has been
>   run on real IAM data yet.
> - **§4(c) generation/rendering:** alphabets are now `dict[str, list[GlyphBitmap]]`. The renderer scales every glyph to a
>   common x-height (`renderer.target_x_height_px`) and places by baseline (descenders drop below the line). The baseline
>   generator no longer forces every letter into one box. The neural path wraps 28x28 output with prior-derived metrics.
>   Old saved alphabets (no `metrics` in `glyph_metadata.json`) are rejected with a clear error.
> - **§5 config:** the seven unread keys are deleted (`preprocessing.gaussian_blur_kernel`, `segmentation.method`,
>   `recognition_training.emnist_split`, `features.glyph_canvas_size`, `training.dataset`, `training.emnist_split`,
>   `generation.jitter_position_px`); new blocks `data.iam`, `evaluation`, `glyph_metrics`; new keys
>   `preprocessing.deskew_min_chars`, `deskew_angle_step_deg`, `segmentation.stack_*`, `renderer.target_x_height_px`.
>   `deskew_angle_search_deg` is now a true search range; `renderer.char_spacing_px` is extra tracking on top of each glyph's advance.
> - **§6 results:** `scripts/evaluate.py` is rewritten: CER (CNN legibility) + writer-embedding cosine distance (style
>   fidelity; **the writer-ID model is a flagged stub**), persisted under `results/<utc>_<sha>/metrics.json` and
>   `results/index.csv`. The numbers in §6 predate this and were computed with the leaky leave-one-out; do not compare.
> - **§7 bugs fixed:** the leave-one-out leak; the `mean_char_width` height bug and the fixed-box/no-descender renderer;
>   `_merge_smallest_gaps` image/bbox mismatch; stroke-width overflow on solid crops; the deskew bias (`minAreaRect` ->
>   projection profile, plus a `deskew_min_chars` gate) and the split i-dot (new `_merge_stacked`).
> - **Still true:** the neural generator is undertrained and the CNN is a 2-epoch model. **Open, verified bug, not fixed:**
>   `_apply_shear` has the opposite sign to `_slant_deg` (a right-leaning stroke measures +20.6 deg; `_apply_shear(+20)` yields a
>   glyph measuring -20.1 deg), and the reference font is itself slanted, so a correct fix is shear by (writer - font) slant.

## 1. Overview

This is a **handwriting synthesis** system, not a handwriting recognition system. Given a photo of one short
handwritten word plus the word typed by the user (e.g. "quick"), it segments the word into letter crops,
measures style (slant, stroke width, height, spacing), builds a full a-z alphabet in that style, and renders
arbitrary lowercase text from that alphabet as a PNG. Unseen letters come from one of two generators:
a non-learned baseline (a Windows system font warped to the measured style; `src/generator/baseline_generator.py`)
or a conditional VAE trained on EMNIST (`src/style_encoder/`).

Recognition is a supporting component, not the product. The default recognition backend (`labeled`) does no
recognition: it trusts the typed word. A small CNN letter classifier (`src/recognition/`) exists to (a) cross-check
the typed word in the UI, (b) optionally replace the typed word ("cnn" backend, experimental), and (c) act as an
"is this generated glyph readable" scorer in `scripts/evaluate.py`. Nothing in the repo recognizes full words or
lines (no CTC/CRNN, no CER/WER). The only handwriting samples in the repo are two **synthetic, font-rendered**
images; the system has never been evaluated on real handwriting.

---

## 2. Directory tree

Excluded: `venv/`, `__pycache__/`, `.pytest_cache/`, `data/raw/hf_cache/` (HF dataset cache), and the contents of
`models/checkpoints/` (`char_cnn.pt`, `style_vae.pt`) and `outputs/` (generated artefacts).

```
handwriting matching/
├── AGENTS.md                      WRONG PROJECT (routing). Ignore.
├── README.md                      Overview/usage/results; partly stale (see §7)
├── PROJECT_STATUS.md              This file
├── text.md                        1270-line original build brief/prompt the project was generated from
├── requirements.txt               Dependencies, lower-bounded, no pins
├── .gitignore                     Ignores venv, *.pt/*.pth checkpoints, data/raw|processed, outputs/*, .streamlit
├── config/
│   └── config.yaml                All tunables (dumped and explained in §5)
├── app/
│   ├── __init__.py                empty
│   └── app.py                     Streamlit UI: upload -> preview -> CNN cross-check -> style metrics -> generate -> render -> download
├── src/
│   ├── pipeline.py                Orchestration: analyze_sample, recognize_with_cnn, generate_alphabet_baseline, generate_alphabet_neural
│   ├── preprocessing/pipeline.py  Preprocessor: resize, gray, denoise, CLAHE, binarize, deskew, crop
│   ├── segmentation/segmenter.py  Segmenter: connected components, dot merge, touching-glyph split, label alignment
│   ├── recognition/
│   │   ├── __init__.py            get_recognizer() factory ("labeled" | "template" | "cnn")
│   │   ├── base.py                Recognizer abstract base class
│   │   ├── labeled.py             LabeledRecognizer: zips glyphs to the typed string
│   │   ├── template_matching.py   TemplateMatchingRecognizer: font-rendered templates + Pearson correlation
│   │   ├── cnn_model.py           CharCNN architecture
│   │   └── cnn_recognizer.py      CNNRecognizer wrapper, normalize_glyph_for_cnn, checkpoint cache
│   ├── features/
│   │   └── style_extractor.py     StyleFeatureExtractor -> GlyphFeatures / StyleProfile (slant, stroke, curvature, ...)
│   ├── generator/
│   │   ├── baseline_generator.py  BaselineGlyphGenerator (font glyph warped to writer style)
│   │   └── alphabet_builder.py    AlphabetBuilder.save/load: PNGs + glyph_metadata.json under outputs/alphabets/<session>/
│   ├── style_encoder/
│   │   ├── model.py               StyleEncoder, Decoder, HandwritingStyleVAE, vae_loss
│   │   └── inference.py           StyleEncoderInference: few-shot latent averaging + glyph generation
│   ├── renderer/
│   │   └── text_renderer.py       TextRenderer: placement, jitter, word wrap, ink colouring
│   ├── evaluation/
│   │   └── metrics.py             ssim_against_reference, character_recognition_accuracy, writer_style_similarity
│   └── utils/
│       ├── config.py              load_config (lru_cached), resolve_path, PROJECT_ROOT
│       ├── emnist_source.py       EMNISTLettersHF dataset from HF Hub `tanganke/emnist_letters` + orientation fix
│       └── logging_setup.py       get_logger (stdout handler)
│   (all package `__init__.py` files are empty except src/recognition/__init__.py)
├── scripts/
│   ├── train_style_encoder.py     Trains the VAE -> models/checkpoints/style_vae.pt
│   ├── train_cnn_recognizer.py    Trains CharCNN with augmentation -> models/checkpoints/char_cnn.pt
│   ├── preprocess_dataset.py      Downloads EMNIST, writes data/processed/emnist_contact_sheet.png, logs class balance
│   ├── generate_alphabet.py       CLI: image + label -> outputs/alphabets/<session>/
│   ├── generate_text.py           CLI: saved alphabet + text -> PNG
│   ├── evaluate.py                Baseline-vs-neural comparison (the only place metrics are computed)
│   └── make_synthetic_sample.py   Renders data/samples/quick_sample.png and hello_sample.png from Windows fonts
├── tests/                         47 pytest tests
│   ├── test_end_to_end.py         (2)  image -> alphabet -> rendered text, baseline backend only
│   ├── test_features.py           (5)
│   ├── test_generator.py          (4)
│   ├── test_preprocessing.py      (6)
│   ├── test_recognition.py        (10) labeled/template/CNN wrappers; 1 test skipped-if no checkpoint
│   ├── test_renderer.py           (5)
│   ├── test_segmentation.py       (8)
│   └── test_style_encoder_model.py (7) architecture only (shapes, loss finite, gradients)
├── docs/
│   ├── architecture.md, methodology.md, experiments.md, limitations.md, future_work.md
│   └── development_log.md         Chronological build log; source of the recorded training numbers
├── data/
│   ├── samples/                   quick_sample.png, hello_sample.png (synthetic)
│   ├── raw/                       HF dataset cache in raw/hf_cache/; raw/EMNIST/raw/ exists but is empty and unused by code
│   └── processed/                 empty (.gitkeep)
├── models/checkpoints/            char_cnn.pt, style_vae.pt (gitignored)
├── outputs/alphabets/, outputs/generated_text/   generated artefacts (gitignored; contain earlier demo runs)
└── notebooks/                     empty
```

---

## 3. Stack

- **Language:** Python 3.12.3 (from the existing `venv/`). README states 3.12. No `pyproject.toml`, `setup.py`,
  `environment.yml`, `package.json` or lock file: **not present**. Modules are imported by putting the repo root
  on `sys.path` (each script and `app/app.py` does `sys.path.insert(0, <repo root>)`); tests are run from the repo root.
- **UI framework:** Streamlit. **Training framework:** PyTorch, CPU (scripts pick CUDA if available; never used).
- **Config:** one YAML file, `config/config.yaml`, loaded by `src/utils/config.py::load_config`.

`requirements.txt` (required lower bound) versus what is installed in `venv/`:

| Package | requirements.txt | Installed |
|---|---|---|
| numpy | >=2.0 | 2.5.3 |
| scipy | >=1.16 | 1.18.1 |
| pillow | >=12.0 | 12.3.0 |
| opencv-python-headless | >=4.10 | 5.0.0.93 |
| scikit-image | >=0.24 | 0.26.0 |
| torch | >=2.14 | 2.14.0 |
| torchvision | >=0.19 | 0.29.0 |
| datasets | >=3.0 | 5.0.1 |
| streamlit | >=1.38 | 1.63.0 |
| PyYAML | >=6.0 | 6.0.3 |
| tqdm | >=4.66 | 4.70.1 |
| scikit-learn | >=1.8 | 1.9.1 |
| pytest | >=8.0 | 9.1.1 |

`scipy`, `scikit-learn`, `tqdm` and `torchvision` are listed, but no import of them was found in `src/`, `scripts/`
or `app/` (`tests/` not checked).

### Model architectures in the repo

1. **`CharCNN`** (`src/recognition/cnn_model.py`), a discriminative classifier. Input 1x28x28. Conv3x3(1->32)+BN+ReLU,
   Conv3x3(32->32)+BN+ReLU, MaxPool2 (->14x14), Conv3x3(32->64)+BN+ReLU, Conv3x3(64->64)+BN+ReLU, MaxPool2 (->7x7),
   Flatten (3136), Linear(3136->128), ReLU, Dropout(0.3), Linear(128->26).
2. **`StyleEncoder`** (`src/style_encoder/model.py`). Input 1x28x28. Conv3x3 s2 p1 (1->32, 14x14), ReLU,
   (32->64, 7x7), ReLU, (64->128, 4x4), ReLU, flatten (2048), then two Linear(2048->latent_dim): `fc_mu`, `fc_logvar`.
   It never receives the class label.
3. **Content embedding**: `nn.Embedding(26, 16)` inside `HandwritingStyleVAE.content_embedding`.
4. **`Decoder`**. Input [z(32); embedding(16)] = 48. Linear(48->2048), reshape 128x4x4,
   ConvTranspose2d 128->64 (k3 s2 p1 op0: 4->7), ReLU, 64->32 (op1: 7->14), ReLU, 32->1 (op1: 14->28), sigmoid.
5. **`HandwritingStyleVAE`**: encoder + embedding + decoder = a conditional VAE. The architecture is hard-wired to
   28x28 input (flatten size 128*4*4).
6. Non-neural "models": `TemplateMatchingRecognizer` (font templates + Pearson correlation) and
   `BaselineGlyphGenerator` (font glyph + shear/stroke/resize).

Parameter counts: not computed.

### Dataset

- **EMNIST-letters** (26 case-merged classes, 28x28 grayscale), loaded by `src/utils/emnist_source.py` from the
  Hugging Face Hub dataset **`tanganke/emnist_letters`** via `datasets.load_dataset`, cached in `data/raw/hf_cache/`
  (present on disk). The first run needs internet. The code comments say torchvision's own EMNIST downloader was
  avoided because its host had an expired TLS certificate.
- Only the `train` split is ever loaded. Full train size: `scripts/train_style_encoder.py`'s docstring says ~145k;
  the auditor did not verify.
- No other dataset is present. No writer-labelled data (IAM etc.) is used; EMNIST has no writer IDs.

---

## 4. Pipelines

### 4(a) Style VAE

| Stage | File :: function | Detail |
|---|---|---|
| Data loading | `src/utils/emnist_source.py :: EMNISTLettersHF.__getitem__` | PIL image -> grayscale -> `_fix_emnist_orientation` (rotate -90, flip left-right) -> float32 /255 -> tensor [1,28,28], ink white (1.0) on black. Label 0..25 = a..z. |
| Subsetting/splitting | `scripts/train_style_encoder.py :: get_dataloaders` | `torch.randperm` (seed `training.seed`=42) over the EMNIST train split, keep first `max_train_samples`; first `int(n*val_fraction)` are val, rest train. Config: 6000 -> **5700 train / 300 val**, batch 32 (179 train batches, 10 val). `num_workers=0`. |
| Preprocessing/augmentation | not present | Raw EMNIST tensors go straight to the model. No augmentation, no normalisation beyond /255. |
| Model | `src/style_encoder/model.py :: HandwritingStyleVAE` | Shapes in §3. latent_dim 32, content_embed_dim 16. |
| Forward | `HandwritingStyleVAE.forward` | `encoder(x)` -> (mu, logvar); `reparameterize`: z = mu + eps*exp(0.5*logvar); embedding of the class index; `decoder(z, content)` -> reconstruction of the same image. |
| Loss | `vae_loss(recon, target, mu, logvar, kl_weight=0.5)` | `BCE(recon, target, sum) / batch  +  0.5 * KL/batch`, where KL = -0.5*sum(1+logvar-mu^2-exp(logvar)) vs N(0,I). `kl_weight=0.5` is a hardcoded default (not in config); the training script does not override it. |
| Training loop | `scripts/train_style_encoder.py :: train` | Adam, lr 1e-3, `epochs` 6, no scheduler, no gradient clipping, no early stopping, no augmentation. Labels are clamped to [0,25]. After every epoch it computes val loss and **saves a checkpoint whenever val loss improves** (dict: `model_state_dict`, `config`, `epoch`, `val_loss`). Logging: stdout only (train loss, recon, KL, val loss, seconds). |
| Val-loss caveat | same | Validation runs `model(...)`, which samples z via `reparameterize`, so val loss is stochastic (not a deterministic posterior-mean evaluation). |
| Sampling / few-shot | `src/style_encoder/inference.py :: StyleEncoderInference` | `encode_writer_style`: `glyph_to_tensor` (square-pad, add `side//5` border, `INTER_AREA` resize to 28) each observed a-z glyph, take encoder **mu**, mean over glyphs -> one writer vector [1,32]. `generate_char`: decoder(writer_vector [+ N(0,sample_noise) jitter], embedding[char]); output thresholded at **0.35** -> uint8 0/255 28x28. `generate_alphabet`: per char `variants` images; variant 0 has no noise, the others get noise 0.05. Sampling from the prior N(0,I) is not present. |
| Integration | `src/pipeline.py :: generate_alphabet_neural` | `variants=3` hardcoded; then overwrites variant 0 of every observed char with the real segmented crop (see the scale bug in §7). |

Checkpoint on disk: `models/checkpoints/style_vae.pt`, epoch 6, val_loss 137.48 (read from the file).

### 4(b) CNN recognizer

- **What it classifies:** a single segmented glyph (28x28) into **26 classes, index 0..25 -> a..z**. EMNIST-letters
  merges upper and lower case, so the label space is really *letter identity, case-agnostic*. It cannot output digits,
  punctuation, or case. The label list is stored in the checkpoint (`labels`).
- **Architecture:** `CharCNN`, §3. **Training:** `scripts/train_cnn_recognizer.py :: train`.
  - Data: the same EMNIST subsetting as the VAE, config `recognition_training` (20000 images, 5% val = 1000, batch 128,
    10 epochs, Adam lr 1e-3, CrossEntropy).
  - Augmentation (`AugmentedEMNIST._augment`, train split only): rotation +/-18 deg, shear +/-14 deg, scale
    0.85-1.15, translation +/-8%, erode or dilate with a 2x2 or 3x3 kernel (p=0.5), Gaussian noise sigma 12 (p=0.5).
  - Checkpoint on best val accuracy: dict `model_state_dict`, `labels`, `num_classes`, `epoch`, `val_acc`.
- **Inference:** `src/recognition/cnn_recognizer.py :: CNNRecognizer.recognize`. Each glyph goes through
  `normalize_glyph_for_cnn` (aspect-preserving resize so the longer side is 0.75*28=21 px, centred on a 28x28 black canvas),
  then softmax; `last_confidences` holds max-probabilities. `_load_checkpoint` is `lru_cache`d.
- **Roles in the system:**
  1. Backend `"cnn"` of `get_recognizer` (`analyze_sample(..., recognition_backend="cnn")`): segmentation is *not*
     forced to the label length and characters come from the CNN. Opt-in, "experimental" in the UI.
  2. Non-destructive cross-check `src/pipeline.py :: recognize_with_cnn` -> `RecognitionCheck` (typed vs predicted,
     confidences, per-glyph agreement, accuracy). Shown in `app/app.py` step 3 ("Recognition check").
  3. OCR-proxy scorer in `scripts/evaluate.py` (`character_recognition_accuracy(..., recognizer=cnn_recognizer)`).
  4. It is **not** on the default generation path (`recognition.backend: "labeled"`).
- Checkpoint on disk: `models/checkpoints/char_cnn.pt`, **epoch 2, val_acc 0.888** (read from the file).

### 4(c) Generation: sample image -> alphabet -> line of text

**Image -> analysis** (`src/pipeline.py :: analyze_sample`)
1. `Preprocessor.process` (`src/preprocessing/pipeline.py`): resize whole image to height 400 (scale capped at 4.0x
   upward; larger images are downscaled), grayscale, `medianBlur` + `fastNlMeansDenoising(h=7,7,21)`, CLAHE
   (clip 2.5, 8x8), `binarize` = adaptive Gaussian threshold AND Otsu then a 2x2 elliptical open (ink=255),
   `estimate_skew_angle` (`minAreaRect` over all ink; if |angle|>15 deg no correction), `deskew`, `crop_to_content` (10 px margin).
2. `Segmenter.segment` (`src/segmentation/segmenter.py`): `_connected_components` (8-connectivity, drop area<20) ->
   `_merge_diacritics` (components shorter than 0.5*median height that overlap a body horizontally and sit within a
   vertical gap window are merged, so i/j dots and t crossbars rejoin) -> `_split_touching_glyphs` (components wider
   than 1.6x an expected char width are cut at valleys of the smoothed column-ink profile, cut count = round(width/expected width))
   -> sort left to right. `baseline_y` = median glyph bottom; `x_height_top_y` = median glyph top.
3. If backend is `labeled` (default): `Segmenter.segment_and_align_to_label` forces the glyph count to `len(label)` by
   `_merge_smallest_gaps` (too many) or `_split_widest` (too few; splits at the midpoint).
4. `LabeledRecognizer` assigns the typed characters by order (raises if counts differ).
5. `observed_glyphs[char.lower()]` = first occurrence's crop (so "hello" gives 4 entries, not 5).
6. `StyleFeatureExtractor.extract_style_profile`: per glyph height, width, aspect, PCA slant, stroke width
   (75th percentile of the distance transform x2), ink density, isoperimetric "curvature"; averaged into a `StyleProfile`;
   char spacing = mean gap between bounding boxes.

**Analysis -> alphabet**
- *Baseline* (`BaselineGlyphGenerator.generate_alphabet`): observed char -> `generate_observed` = the real crop resized to
  the writer's mean height, own aspect kept (1 variant). Unobserved char -> `generate_unobserved`: render the char in a
  Windows font at 200 px (`_render_reference_glyph`), resize to **(mean_height x mean_height*mean_aspect)**, dilate/erode
  by `round(|ratio-1|*3)` iterations of a 3x3 ellipse (`_adjust_stroke_width`), shear by tan(mean slant) (`_apply_shear`),
  resize back, rotate by uniform(+/-1.5 deg) with RNG seed 42 (`_jitter`); 3 variants. **Every synthesized letter is forced
  into the same box** (verified: 'a', 'b', 'm' are all 94x62 for `quick_sample.png`), so there is no x-height/ascender
  distinction and no narrow 'i' or wide 'm'.
- *Neural*: §4(a). Generated glyphs are **28x28** and are not rescaled; observed glyphs overwrite variant 0 at their
  original crop size. Verified on `quick_sample.png`: observed q/u/i/c/k are 119x74, 80x70, 73x29, 77x59, 120x78 while
  generated a/b/m are 28x28. The renderer does not rescale, so a neural alphabet rendered as-is mixes ~75-120 px letters
  with 28 px letters.
- *Persistence*: `AlphabetBuilder.save` writes `outputs/alphabets/<session>/lowercase/<c>.png`, `<c>_variant<N>.png` and
  `glyph_metadata.json` (backend, observed/generated chars, per-glyph size and `observed` flag); `load` reads them back.

**Alphabet -> line of text** (`src/renderer/text_renderer.py :: TextRenderer`)
- Input text is lower-cased by callers (`app/app.py`, `scripts/generate_text.py`); `_pick_glyph` would upscale an
  uppercase char's lowercase glyph by 1.15, but that path is not reachable from the UI/CLI.
- `render_line`: iterate characters. Space advances the cursor by `word_spacing_px` (22). A character with no glyph
  (digits, punctuation) advances by `mean_char_width + char_spacing_px` and draws nothing (one warning per char).
  Otherwise pick a random variant (RNG seed 7), `_jitter_glyph` (rotate +/-`rotation_jitter_deg`=1.0 deg with
  `INTER_NEAREST`; vertical offset integer in +/-`baseline_jitter_px`=2), record x = cursor, advance
  cursor += glyph width + `char_spacing_px` (4).
- Baseline: `baseline_y = canvas_margin_px + max_glyph_height`. Each glyph is placed so its **bottom edge** sits at
  `baseline_y + jitter`, then clamped inside the canvas. **All glyph bottoms share a baseline: descenders (g, j, p, q, y)
  do not drop below it.** Canvas = width `cursor + 2*margin`, height `int(max_h*1.6) + 2*margin` (margin 40).
- Compositing: `np.maximum` of each glyph into a black canvas (ink=255), so overlapping glyphs merge.
- `render_text`: word wrap at `max_line_width_px=1400` (a default argument, not configurable) using an *estimated*
  width `len(line)*(mean_char_width+char_spacing)`; words are never split; lines are rendered independently and stacked
  left-aligned with no extra gap (each line carries its own 2*margin). `\n` is not handled.
- `render_to_bgr_on_white`: alpha-blend ink colour BGR (120,40,20) onto white by pixel intensity.
- Cursive joining, kerning and pen-lift-based spacing: not present.

---

## 5. Config: `config/config.yaml` (full dump)

```yaml
# Central configuration for the handwriting-ai pipeline.
# All modules read tunable parameters from here instead of hard-coding them.

preprocessing:
  target_line_height: 400          # image is upscaled so the text line is this tall (px) before processing
  gaussian_blur_kernel: 3
  adaptive_thresh_block_size: 35
  adaptive_thresh_C: 11
  deskew_angle_search_deg: 15       # search range for deskewing, +/- degrees
  median_filter_kernel: 3

segmentation:
  min_component_area: 20            # px^2, filters out noise specks
  merge_gap_px: 4                   # horizontal gap below which components are merged (dot on i/j, etc.)
  method: "connected_components"    # base method; projection-profile splitting (below) runs on top of it
  split_touching_glyphs: true       # detect over-wide connected components (touching/cursive-joined
                                     # letters) and split them at projection-profile valleys
  wide_component_width_ratio: 1.6   # a component wider than this * expected char width is treated as touching glyphs
  min_split_segment_px: 6           # minimum width (px) for a split-off sub-glyph, else the split is discarded

recognition:
  # Recognition backend is pluggable (see src/recognition/base.py).
  # "labeled" = user supplies ground-truth text for the sample (most reliable when correct).
  # "template" = nearest-neighbour template matching baseline.
  # "cnn" = trained CharCNN classifier (src/recognition/cnn_recognizer.py) — actually looks at
  #   the pixels; used both as a selectable backend and as an always-on cross-check in the app.
  backend: "labeled"
  cnn_checkpoint_path: "models/checkpoints/char_cnn.pt"

recognition_training:
  # Training config for the CNN recognizer (scripts/train_cnn_recognizer.py),
  # analogous to `training:` below but for the classifier, not the style VAE.
  emnist_split: "letters"
  batch_size: 128
  epochs: 10
  lr: 0.001
  val_fraction: 0.05
  seed: 42
  max_train_samples: 20000          # cap for CPU-feasible training time; raise for a better final run
  augment: true                     # random affine + erosion/dilation + noise, for robustness to
                                     # slanted, variable-stroke-width, cursive-influenced letterforms
  checkpoint_path: "models/checkpoints/char_cnn.pt"

features:
  glyph_canvas_size: 64             # normalized glyph square size (px) used for feature extraction & the encoder

style_encoder:
  latent_dim: 32
  content_embed_dim: 16
  image_size: 28                    # matches EMNIST
  num_classes: 26                   # lowercase a-z

training:
  dataset: "EMNIST"
  emnist_split: "letters"
  batch_size: 32
  epochs: 6
  lr: 0.001
  val_fraction: 0.05
  seed: 42
  max_train_samples: 6000           # cap for CPU-feasible training time / this dev machine is currently very low on free RAM
  checkpoint_path: "models/checkpoints/style_vae.pt"

generation:
  variants_per_char: 3
  jitter_position_px: 2
  jitter_rotation_deg: 1.5

renderer:
  char_spacing_px: 4
  word_spacing_px: 22
  line_height_multiplier: 1.6
  baseline_jitter_px: 2
  rotation_jitter_deg: 1.0
  canvas_margin_px: 40
```

### What each key does in code

Legend: **UNUSED** = grepped `src/`, `scripts/`, `app/`, `tests/`; no code reads it.

| Key | Read by | Actual effect |
|---|---|---|
| `preprocessing.target_line_height` (400) | `Preprocessor._resize_to_target_height` | Scales the **whole image** to this height (upscale capped at 4x; large images are also downscaled). The comment says "text line"; it is image height. |
| `preprocessing.gaussian_blur_kernel` | **UNUSED** | Denoising uses `median_filter_kernel` and `fastNlMeansDenoising`, no Gaussian blur. |
| `preprocessing.adaptive_thresh_block_size` (35) | `Preprocessor.binarize` | Adaptive-threshold window (forced odd). |
| `preprocessing.adaptive_thresh_C` (11) | `Preprocessor.binarize` | Constant subtracted in the adaptive threshold. |
| `preprocessing.deskew_angle_search_deg` (15) | `Preprocessor.estimate_skew_angle` | Not a search: if the `minAreaRect` angle exceeds this, the angle is set to 0 (no deskew). |
| `preprocessing.median_filter_kernel` (3) | `Preprocessor.denoise` | `cv2.medianBlur` kernel. |
| `segmentation.min_component_area` (20) | `Segmenter._connected_components` | Drops smaller components. |
| `segmentation.merge_gap_px` (4) | `Segmenter._merge_diacritics` | **Vertical** gap window (-gap .. 4*gap) for attaching dots to bodies. The comment says "horizontal gap". |
| `segmentation.method` | **UNUSED** | Only connected components exist. |
| `segmentation.split_touching_glyphs` (true) | `_split_touching_glyphs` | Master switch for width-based splitting. |
| `segmentation.wide_component_width_ratio` (1.6) | `_split_touching_glyphs` | Components wider than ratio x expected width get split. |
| `segmentation.min_split_segment_px` (6) | `_split_touching_glyphs`, `_projection_profile_splits` | Minimum piece width, cut-search margin, and eligibility (`w > 2*min_w`). |
| `recognition.backend` ("labeled") | `pipeline.analyze_sample` | Default backend; also selects the segmentation path (`labeled` -> align to label length, anything else -> raw `segment`). |
| `recognition.cnn_checkpoint_path` | `pipeline.py`, `app/app.py`, `scripts/evaluate.py` | CNN checkpoint location (duplicated as `DEFAULT_CHECKPOINT` constant in `cnn_recognizer.py`). |
| `recognition_training.emnist_split` | **UNUSED** | Dataset id and split are hardcoded in `emnist_source.py`. |
| `recognition_training.batch_size/epochs/lr/val_fraction/seed/max_train_samples/augment/checkpoint_path` | `scripts/train_cnn_recognizer.py` | As named. `seed` seeds torch, python `random`, the subset permutation and the augmentation RNG. |
| `features.glyph_canvas_size` (64) | **UNUSED** | No 64-px canvas is used in feature extraction or the encoder (metrics.py hardcodes its own 64x64 for SSIM). |
| `style_encoder.latent_dim`, `content_embed_dim`, `num_classes` | `train_style_encoder.py`, `StyleEncoderInference` | Model sizes. Changing `latent_dim`/`content_embed_dim` requires retraining. |
| `style_encoder.image_size` (28) | `StyleEncoderInference.__init__` -> `glyph_to_tensor` only | Not used in training; the model architecture is hard-wired to 28x28, so changing it breaks inference. |
| `training.dataset` | **UNUSED** | |
| `training.emnist_split` | **UNUSED** | |
| `training.batch_size/epochs/lr/val_fraction/seed/max_train_samples` | `scripts/train_style_encoder.py` | As named. |
| `training.checkpoint_path` | `train_style_encoder.py`, `StyleEncoderInference`, `app/app.py`, `evaluate.py` | VAE checkpoint location. |
| `generation.variants_per_char` (3) | `BaselineGlyphGenerator.__init__` | **Baseline only.** Neural path hardcodes `variants=3` in `pipeline.generate_alphabet_neural`. |
| `generation.jitter_position_px` | **UNUSED** | |
| `generation.jitter_rotation_deg` (1.5) | `BaselineGlyphGenerator.__init__` | Baseline only. |
| `renderer.char_spacing_px` (4) | `TextRenderer.render_line`, `render_text` | Gap added after every drawn glyph; also in the wrap estimate. |
| `renderer.word_spacing_px` (22) | `render_line` | Cursor advance for a space. |
| `renderer.line_height_multiplier` (1.6) | `render_line` | Canvas height = int(max_h * 1.6) + 2*margin. |
| `renderer.baseline_jitter_px` (2) | `_jitter_glyph` | Random vertical offset +/-2 px per glyph. |
| `renderer.rotation_jitter_deg` (1.0) | `_jitter_glyph` | Random rotation +/-1 deg per glyph. |
| `renderer.canvas_margin_px` (40) | `render_line` | Margin around each line's canvas. |

Read nowhere: `preprocessing.gaussian_blur_kernel`, `segmentation.method`, `recognition_training.emnist_split`,
`features.glyph_canvas_size`, `training.dataset`, `training.emnist_split`, `generation.jitter_position_px`.

---

## 6. Results

### Metrics the code computes

| Metric | Computed in | How | Data |
|---|---|---|---|
| Val loss (BCE + 0.5*KL) | `scripts/train_style_encoder.py :: train`, logged per epoch and stored in the checkpoint | Mean over 10 val batches; z is sampled, so it is noisy | 300 EMNIST-train images not used for VAE training |
| Val accuracy (top-1, 26-way) | `scripts/train_cnn_recognizer.py :: _accuracy`, logged per epoch and stored in the checkpoint | argmax vs label, **unaugmented** images | 1000 EMNIST-train images not used for CNN training |
| Leave-one-out SSIM | `scripts/evaluate.py :: leave_one_out_ssim` + `src/evaluation/metrics.py :: ssim_against_reference` | For each distinct observed letter, regenerate it and compare to the real crop; both resized to 64x64 (`INTER_AREA`), `skimage.structural_similarity(data_range=255)`; mean over letters | The single sample image (5 letters for "quick") |
| OCR-proxy accuracy | `metrics.py :: character_recognition_accuracy` | Fraction of 26 letters where recognizer(`alphabet[c][0]`) == c. Run with the template recognizer, and with the CNN if a checkpoint exists. **Includes the observed letters (real pixels).** With n=26, one letter = 3.85 points. | The generated alphabet of one sample |
| Writer-style similarity | `metrics.py :: writer_style_similarity` | Cosine similarity of 6-number vectors (mean aspect, mean slant, mean stroke width, mean ink density, mean curvature, std slant) between the 26-glyph generated alphabet and the observed glyphs. Features are **not scaled**, so slant and stroke width dominate. | The generated alphabet vs the same sample |
| CNN/typed agreement | `pipeline.py :: recognize_with_cnn` (`RecognitionCheck.accuracy`), shown in the app | Fraction of segmented glyphs where CNN prediction == typed char | The uploaded sample |
| Human evaluation | Protocol only in `docs/experiments.md` | **Not present / not run** | n/a |
| CER, WER | **Not present** | | |

### Held-out data

- **There is no held-out evaluation set for generation.** Every generation metric is computed on one or two
  synthetic, font-rendered images that were also used to build the alphabet.
- The only held-out data are the 300 (VAE) and 1000 (CNN) validation images, drawn from the EMNIST **train**
  split. The EMNIST **test** split is never loaded. Checkpoint selection and reported numbers use the same
  validation images, so they are slightly optimistic.
- No writer-disjoint split, no real-handwriting evaluation, no repeated runs, no confidence intervals.
- Results are not written to a file. They exist as stdout from `evaluate.py`, in checkpoint metadata, and as prose
  in `README.md` and `docs/development_log.md`.

### Latest numbers

Recorded training results (read from the checkpoint files):
- `models/checkpoints/char_cnn.pt`: epoch 2, **val_acc 0.888**. Training was killed by a system low-memory event after
  epoch 2 of the configured 10 (`docs/development_log.md`); that checkpoint was kept.
- `models/checkpoints/style_vae.pt`: epoch 6 (of 6), **val_loss 137.48**. README says validation loss fell 208.3 -> 137.5.
  An earlier 8-epoch/40k attempt was killed by low memory (dev log) and the config was cut to 6000 images/6 epochs.

Baseline vs neural, from `scripts/evaluate.py`, **re-run 2026-09-20** (synthetic samples, checkpoints as above):

| Sample | Metric | Baseline | Neural (VAE) |
|---|---|---|---|
| quick | OCR proxy, template | 30.8% | 11.5% |
| quick | OCR proxy, CNN | 57.7% | 19.2% |
| quick | Writer-style similarity | 0.786 | 0.791 |
| quick | LOO SSIM (mean) | 0.203 | 0.219 |
| hello | OCR proxy, template | 26.9% | 0.0% |
| hello | OCR proxy, CNN | 38.5% | 15.4% |
| hello | Writer-style similarity | 0.814 | 0.991 |
| hello | LOO SSIM (mean) | 0.182 | 0.156 |

- The "quick" template/similarity/SSIM values match the README table. The CNN-OCR values and everything for "hello"
  are not in the README. On readability the baseline beats the neural generator on both samples.
- **LOO SSIM baseline leak:** `leave_one_out_ssim` builds the baseline glyph with `full_analysis.style_profile`, which
  was computed *including* the held-out letter (its height, width and slant). The neural branch correctly excludes it.
  The two SSIM columns are therefore not a fair comparison.
- README also claims the CNN agreed with the typed label on 4/5 characters for each sample (confidence 0.66-0.99). Not re-run.
- Test suite: `pytest tests/` -> **47 passed in ~12 s** (README's "40/40" is stale).

---

## 7. State: finished, half-finished, broken

### Finished and working (exercised by tests or the audit's runs)
- Preprocessing, segmentation (incl. diacritic merge and touching-glyph split), feature extraction, baseline generation,
  alphabet save/load, text rendering, and the baseline end-to-end path.
- `labeled`, `template` and `cnn` recognizers; `recognize_with_cnn`.
- Both training scripts exist and (per the dev log) ran; both checkpoints exist and load.
- `scripts/evaluate.py`, `generate_alphabet.py` and `generate_text.py` run (evaluate.py was run by the audit).

### Working but weak (blunt)
- **The neural generator is undertrained and its output is not directly usable.** 6000 images, 6 epochs, 28x28 output,
  and mixed glyph scales (below). It scores worse than the baseline on readability.
- **The CNN checkpoint is a 2-epoch model** (10 configured).
- **The default recognition path does no recognition** (`labeled` trusts the typed word).
- **All evidence is synthetic.** No real handwriting sample exists in the repo.
- **Segmentation is heuristic.** Character count for a blob is estimated from width alone, and the label-alignment step
  merges or midpoint-splits blindly. On `quick_sample.png` the segmenter found 6 glyphs for a 5-letter word before alignment.
- **Style-encoder disentanglement is unverified:** EMNIST has no writer IDs, so nothing checks that the latent encodes
  writer style rather than generic shape variation (`docs/limitations.md` says the same).

### Not present
CER/WER, word- or line-level recognition (CRNN/CTC), writer-labelled training or writer-identity loss, uppercase/digit/punctuation
glyphs, cursive joining in the renderer, descender handling, vector/stroke generation, SVG/font export, human evaluation,
test-split evaluation, CI, lint config, git history, any tests for `scripts/*`, `app/app.py`, `src/evaluation/metrics.py`,
`AlphabetBuilder`, `emnist_source.py`, or `StyleEncoderInference` (VAE architecture is tested; inference is not).

### Known bugs (verified by reading code and/or running it)
1. **Mixed glyph scales in the neural alphabet** (`src/pipeline.py :: generate_alphabet_neural`): observed crops at native
   size (e.g. 119x74) sit beside generated 28x28 glyphs; `TextRenderer` does not rescale. Reproduced on `quick_sample.png`.
   The OCR-proxy metrics normalise size, so they hide this.
2. **Baseline forces all letters into one box** (`BaselineGlyphGenerator.generate_unobserved`): identical width and height for
   every synthesized letter.
3. **`TextRenderer.__init__` width bug:** `widths = [imgs[0].shape[0] ...]` reads the height, so `mean_char_width` is actually
   mean height. It affects blank-glyph advance and the wrap estimate.
4. **`Segmenter._merge_smallest_gaps`** sets the merged glyph's `image` to the larger of the two crops instead of the union,
   although the bbox covers both.
5. **No descenders:** every glyph's bottom edge is aligned to the baseline (g, j, p, q, y sit on it).
6. **Baseline LOO SSIM leaks the held-out glyph** into the style profile (§6).
7. **Unread config keys** (§5).
8. **Dead/misleading code and comments:** unused `mask` variable in `render_to_bgr_on_white`; `baseline_generator.py` docstring
   references nonexistent `src/generator/conditional_generator.py`; `_stroke_width` comment describes local maxima but the code
   uses a percentile; config comments for `target_line_height` and `merge_gap_px` misdescribe behaviour.
9. **Stale docs:** README Training section says the VAE trains on 40k images for 8 epochs (~20-25 min); config says 6000/6/batch 32.
   README test count 37+/40 vs actual 47. `AGENTS.md` is a different project entirely.
10. **Unexplained observation:** `evaluate.py` logs a deskew of -11.35 deg on `quick_sample.png` and 6 raw glyphs for 5 letters.
    `make_synthetic_sample.py` applies a 3.5 deg rotation; the cause of the mismatch was not investigated.
11. **Backend-name inconsistency:** the app stores the radio-button label (e.g. "baseline (style-transformed reference font)")
    as `generator_backend` in `glyph_metadata.json`; the CLI stores "baseline"/"neural".
12. **`TemplateMatchingRecognizer` default charset includes uppercase**, so `get_recognizer("template")` can return capital letters.
13. **Label sanitising is done by callers**, not by `analyze_sample`: a label with non-letters or a glyph-count mismatch raises
    in `LabeledRecognizer`.

### Hardcoded paths and environment coupling
- Windows font paths `C:/Windows/Fonts/{segoesc,comic,arial,segoepr}.ttf` in `baseline_generator.py`, `template_matching.py`,
  `make_synthetic_sample.py`, `tests/test_end_to_end.py`. On failure it falls back to PIL's default bitmap font (untested
  here); the e2e test skips if no fonts are found.
- `outputs/alphabets`, `outputs/generated_text`, `data/raw/hf_cache`, `data/processed`, `data/samples` are hardcoded in scripts/app.
- `HF_DATASET_ID = "tanganke/emnist_letters"` hardcoded in `src/utils/emnist_source.py`.
- README commands use `venv\Scripts\...` (Windows).
- Dev machine: 4 cores, ~8 GB RAM, repeatedly hit low memory during training (`docs/development_log.md`).

### Magic numbers (not in config)
- Preprocessing: CLAHE clip 2.5 / tiles 8x8; `fastNlMeansDenoising(h=7, template 7, search 21)`; 2x2 open kernel; crop margin 10; upscale cap 4.0.
- Segmentation: diacritic threshold 0.5 x median height; vertical window (-gap, 4*gap); cut margin `max(min_w, 0.3*expected_w)`;
  3-tap smoothing kernel; expected width fallback 0.75 x median height; "normal" width <= 1.3 x median height.
- CNN: augmentation ranges (§4b); `normalize_glyph_for_cnn` margin 0.75.
- VAE: `kl_weight=0.5`; threshold 0.35; `sample_noise=0.05`; `glyph_to_tensor` border `side//5`; `variants=3` in the neural path.
- Baseline: font size 200, RNG seed 42, stroke iterations `round(|ratio-1|*3)`, shear `tan(mean slant)`.
- Renderer: RNG seed 7, `max_line_width_px=1400`, upscale 1.15 for capitals, ink colour BGR (120, 40, 20), alpha threshold `>20` (unused).
- Metrics: SSIM resize 64x64; 6-feature style vector; OCR proxy evaluates the first variant only.

### Fragile
- `load_config` is `lru_cache`d: config edits need a process restart.
- `torch.load(..., weights_only=False)` in `StyleEncoderInference` executes pickled data from the checkpoint file.
- `StyleEncoderInference` reloads the checkpoint each time it is constructed (`evaluate.py`'s leave-one-out loop does this per letter).
- **Re-running a training script overwrites the existing checkpoint** on its first epoch (best value starts at `inf` for the VAE
  and 0.0 for the CNN). Back up `models/checkpoints/*.pt` first.
- The pipeline assumes lowercase a-z, a single text line, ink-dark-on-light-paper input, and a word with at least 1 letter.
- `sys.path.insert` bootstrapping instead of a package install.

---

## 8. Commands

Run from the repo root (`handwriting matching/`). The venv already exists with everything installed.

```powershell
# Setup (only if starting fresh)
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# Tests (47 tests, ~12 s)
venv\Scripts\python -m pytest tests/ -v

# Optional: create the synthetic sample images (writes data\samples\quick_sample.png, hello_sample.png)
venv\Scripts\python scripts\make_synthetic_sample.py

# Optional dataset sanity check (needs internet on first run; writes data\processed\emnist_contact_sheet.png)
venv\Scripts\python scripts\preprocess_dataset.py

# Training (needs internet on first run for EMNIST; hyperparameters in config\config.yaml)
venv\Scripts\python scripts\train_style_encoder.py      # VAE  -> models\checkpoints\style_vae.pt   (training: block)
venv\Scripts\python scripts\train_cnn_recognizer.py     # CNN  -> models\checkpoints\char_cnn.pt    (recognition_training: block)

# Generation (CLI)
venv\Scripts\python scripts\generate_alphabet.py --image data\samples\quick_sample.png --label quick --backend baseline --session-id demo
venv\Scripts\python scripts\generate_alphabet.py --image data\samples\quick_sample.png --label quick --backend neural --session-id demo_nn
#   optional: --checkpoint <path to a VAE .pt>
venv\Scripts\python scripts\generate_text.py --session-id demo --text "the quick fox" --out outputs\generated_text\demo.png

# Evaluation (prints to stdout only; writes no files)
venv\Scripts\python scripts\evaluate.py --image data\samples\quick_sample.png --label quick
venv\Scripts\python scripts\evaluate.py --image data\samples\hello_sample.png --label hello
venv\Scripts\python scripts\evaluate.py --image data\samples\quick_sample.png --label quick --skip-neural

# UI
venv\Scripts\streamlit run app\app.py
```

Notes:
- The neural backend and the "Recognition check" panel require `style_vae.pt` / `char_cnn.pt` respectively; the app
  disables or hides them when the file is missing.
- `scripts/generate_text.py` lower-cases the text and can only render characters present in the saved alphabet (a-z);
  digits and punctuation become blank gaps.
- Any non-CLI use of the pipeline: `from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural`
  and `from src.renderer.text_renderer import TextRenderer` (see `tests/test_end_to_end.py` for a working call sequence).
