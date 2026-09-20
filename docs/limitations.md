# Limitations

Documented explicitly per the design brief's instruction not to overstate
what was built. Grouped by pipeline stage.

## Preprocessing (src/preprocessing/pipeline.py)
- Deskew uses `cv2.minAreaRect` over all ink pixels, which conflates true
  photo tilt with the handwriting's own intrinsic slant. It corrects
  gross page rotation well but is not a precise per-line baseline
  straightener.
- Binarization (adaptive threshold + Otsu AND) is tuned for reasonably
  even lighting; very strong shadows or colored paper can still degrade
  segmentation quality.

## Segmentation (src/segmentation/segmenter.py)
- Connected-component segmentation assumes characters are mostly
  separate ink blobs. It correctly reunites split characters (dot on
  i/j) via a proximity-merge heuristic. It used to unconditionally
  **under-segment** true cursive script where adjacent letters share a
  continuous stroke; `_split_touching_glyphs` now detects components much
  wider than an expected single-character width (including the
  degenerate case of an entire cursive word as one component) and splits
  them at projection-profile valleys. This is a heuristic, not a learned
  segmentation model: it assumes touching letters still thin out where
  one stroke hands off to the next, and it estimates how many characters
  a wide blob contains from width alone — genuinely ambiguous or heavily
  overlapping cursive (no thin handoff point, or an atypical width/count
  ratio) can still be misattributed. See docs/future_work.md for why a
  learned segmentation network or segmentation-free HTR remains the more
  robust long-term fix.
- The `segment_and_align_to_label` reconciliation step (merging the
  smallest gaps or splitting the widest component to match the expected
  character count) is a practical patch, not a real segmentation
  algorithm, and is only used by the `labeled` recognition backend —
  it depends on the user supplying an accurate ground-truth label for
  their own sample. The `cnn` backend instead calls plain `segment()`
  (with the touching-glyph splitter above) and does not force a
  particular glyph count.

## Recognition (src/recognition/)
- The `labeled` backend (default) is not a solution to handwritten
  character recognition — it trusts the user's stated ground truth and
  relies on segmentation being correct. This remains the default because
  the alphabet-generation pipeline needs *correct* per-character identity
  to build a usable personalized alphabet, and a misrecognized character
  mis-keys `observed_glyphs` (see docs/methodology.md for the original
  scope rationale).
- The `template` backend (non-learned, font-template nearest-neighbor) is
  provided as a baseline; it is measurably less accurate than the CNN
  backend and is not used by default.
- The `cnn` backend (`src/recognition/cnn_recognizer.py`, trained via
  `scripts/train_cnn_recognizer.py`) is a real learned classifier —
  validation accuracy on held-out EMNIST-letters is logged at the end of
  training (see the training log / `models/checkpoints/char_cnn.pt`'s
  stored `val_acc`). It is trained on isolated, augmented EMNIST glyphs,
  not on real handwritten photos or true unconstrained cursive text, so
  expect lower accuracy on: unusual/idiosyncratic letterforms not well
  represented in EMNIST's style diversity, and segmentation artifacts
  (an under- or over-split glyph is, by construction, not a clean single
  letter, so the classifier is asked an ill-posed question). Use
  `src/pipeline.py::recognize_with_cnn`'s per-glyph confidence to tell
  these cases apart from genuine style-diversity misses. It still
  performs whole-word recognition one segmented character at a time —
  true segmentation-free cursive-word recognition (CRNN+CTC) is future
  work (docs/future_work.md), since it needs either a licensed word-level
  handwriting corpus or a synthetic cursive-font corpus not yet built.

## Style feature extraction (src/features/style_extractor.py)
- Slant is estimated via PCA over ink-pixel coordinates, which is a
  coarse global measure — it doesn't distinguish stem slant from
  bowl/curve slant within a single glyph.
- Stroke width is a percentile of the distance-transform, not a true
  skeleton-based medial-axis measurement, so it can be biased by glyph
  shape (e.g. a glyph with a large enclosed area vs. a thin stem).

## Generation
- **Baseline generator**: purely a deterministic CV transform of a
  neutral reference font; it has no notion of *this specific writer's*
  letterforms beyond four aggregate numbers (height, width, slant,
  stroke width) and cannot reproduce idiosyncratic letter shapes (e.g. a
  distinctive loop or ligature).
- **Neural generator**: trained on EMNIST-letters, which has no writer
  ID metadata per image (see docs/methodology.md for why this dataset
  was chosen anyway). This means the model learns a general, transferable
  notion of "style factors that vary across handwriting" but was never
  explicitly supervised to cluster multiple images from the *same*
  writer together — the few-shot averaging step is a principled
  latent-space operation, but its effectiveness at capturing one writer's
  idiosyncratic identity (vs. generic stylistic variation) has not been
  validated against ground truth from a writer-labeled dataset.
- Both generators currently target lowercase a-z only (per explicit
  project priority). Uppercase falls back to an upscaled lowercase glyph
  in the renderer; digits/punctuation render as blank space. None of
  these are architectural limits — see future_work.md.
- 28x28 EMNIST resolution caps the fine detail the neural generator can
  reproduce; generated glyphs are visibly lower-resolution than the
  baseline generator's font-derived output. This is a deliberate
  MVP-scale tradeoff for CPU-only training speed.

## Evaluation (src/evaluation/metrics.py)
- "Writer style similarity" is a cosine-similarity proxy over rule-based
  features, not a trained writer-verification model — see
  docs/methodology.md for why (no writer-labeled training data used in
  this MVP).
- Character recognition accuracy uses the same non-learned template
  matcher as the recognition baseline, so it inherits that method's
  ceiling — a low score could mean either the generated glyph is
  genuinely poor or that template matching itself is a weak recognizer
  for that letter.
- Human evaluation is specified as a protocol (docs/experiments.md) but
  not automated or run in this session — it needs actual human raters.

## Environment
- Developed and tested on Windows with **no GPU**. All training/inference
  code paths are CPU-only; the model is intentionally small so this is
  workable, but any future architecture upgrade (larger CNN, GAN,
  diffusion) will need a GPU to train in reasonable time (see
  docs/architecture.md, "Hardware-driven choices").
- `torchvision.datasets.EMNIST`'s built-in downloader points at a
  NIST-hosted mirror that has served an expired TLS certificate for an
  extended period. This project does not work around that by weakening
  TLS verification; instead it sources the same academic dataset from a
  Hugging Face Hub mirror over a normally verified connection (see
  `src/utils/emnist_source.py`).
