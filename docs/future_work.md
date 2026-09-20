# Future Work

Explicitly deferred, but designed for (section 31 of the brief) — noting
*where* each would plug into the existing architecture rather than
requiring a rewrite.

## Recognition
- ~~**CNN character classifier**~~ — **done.** `src/recognition/cnn_model.py`
  (`CharCNN`) + `src/recognition/cnn_recognizer.py` (`CNNRecognizer`,
  registered in `get_recognizer` as backend `"cnn"`), trained via
  `scripts/train_cnn_recognizer.py` on EMNIST-letters with random-affine +
  morphological (erosion/dilation) + noise augmentation so it tolerates
  slant and variable stroke width rather than only upright EMNIST-style
  printing. `src/pipeline.py::recognize_with_cnn` runs it as a
  non-destructive cross-check against the "labeled" backend's assumption
  (surfaced in the app as a per-glyph agreement/confidence panel), and
  `analyze_sample(..., recognition_backend="cnn")` lets it drive detection
  end-to-end (open-vocabulary, no typed label required) when segmentation
  isn't forced to a known length. See docs/limitations.md for its accuracy
  and remaining gaps.
- Segmentation-free / attention-based HTR (e.g. CRNN + CTC) for full
  sentences, removing the need for per-character segmentation entirely —
  still the right eventual answer for cursive at scale. What's implemented
  now instead (`Segmenter._split_touching_glyphs`, see below) is a
  projection-profile heuristic that splits over-wide connected components
  at valleys in the vertical ink-density profile, so common touching
  cursive joins no longer collapse into one unrecognizable blob without
  requiring a new training corpus. A true CRNN+CTC model would need
  word-level handwriting data (IAM/CVL — access agreements not yet in
  place, see the Style modeling section below) or a synthetic
  cursive-font-rendered corpus (in the spirit of
  `src/recognition/template_matching.py`'s font-template baseline) to
  train from scratch offline.
- The shipped `models/checkpoints/char_cnn.pt` was trained for only 2 of
  the configured 10 epochs (88.8% val accuracy) before a system-wide
  low-memory kill — same root cause as the VAE's undertraining below, see
  docs/development_log.md's 2026-09-15 entry. Re-run
  `scripts/train_cnn_recognizer.py` for the full 10 epochs (or raise
  `recognition_training.max_train_samples`/`.epochs` further) once more
  RAM is available; the checkpointing-on-improvement logic means this is
  a pure re-run, no code changes needed.

## Segmentation
- ~~**Cursive-joined-letter splitting**~~ — **done**, as a heuristic.
  `Segmenter._split_touching_glyphs` (config keys `segmentation.
  split_touching_glyphs`, `wide_component_width_ratio`,
  `min_split_segment_px`) detects connected components much wider than an
  expected single-character width — including the degenerate case where an
  entire cursive word is one component — and splits them at projection-
  profile valleys. It is still a heuristic, not a learned segmentation
  model: genuinely ambiguous or heavily overlapping cursive (e.g. no valley
  ever gets thin enough) can still be misattributed. A learned
  segmentation network, or the segmentation-free HTR route above, remains
  the more robust long-term fix.

## Style modeling
- Train on a writer-labeled dataset (IAM, CVL) once the necessary access
  agreements are in place, adding a writer-identity auxiliary loss
  (e.g. triplet loss pulling same-writer style latents together) — this
  directly strengthens the few-shot averaging step, which today relies
  on EMNIST's un-labeled-by-writer diversity alone (see
  docs/limitations.md).
- Increase `training.max_train_samples` / `training.epochs` in
  `config/config.yaml` for a longer final training run once a GPU is
  available, and/or increase encoder/decoder channel widths in
  `src/style_encoder/model.py`.
- Upgrade the generator from a VAE to a conditional GAN or diffusion
  model for higher-fidelity, higher-resolution glyphs (see
  docs/architecture.md for why this was deferred past the MVP) — the
  `StyleEncoderInference` interface (`encode_writer_style`,
  `generate_char`) is designed to stay stable across this swap.
- Multi-scale / higher-resolution glyph generation beyond EMNIST's 28x28.

## Charset coverage
- Uppercase A-Z: retrain with `style_encoder.num_classes = 52` (or a
  parallel model) using EMNIST's `byclass`/`balanced` splits, which
  include case; export to `outputs/alphabets/<session>/uppercase/`
  alongside the existing `lowercase/` directory
  (`AlphabetBuilder.save(..., charset_case="uppercase")` already supports
  this).
- Digits 0-9 and basic punctuation, same approach.
- Personalized punctuation, mathematical symbols.

## Rendering / output formats
- Multiple real stroke variants per generated character (not just model
  latent jitter) once more training data per writer is available.
- Handwriting on ruled paper; simulated pen pressure / stroke-trajectory
  generation (a true sequential/stroke-based model, e.g. an RNN over pen
  coordinates, rather than rasterized glyphs — a substantially different
  representation from the current raster-VAE approach).
- Cursive joining between letters (currently each glyph is placed
  independently).
- Export to SVG/PDF, custom TTF/OTF font generation, printer/Word/Notepad
  integration.

## Evaluation
- Real LPIPS perceptual metric (currently SSIM only — LPIPS needs a
  pretrained perceptual network, deferred to keep dependencies light for
  the MVP).
- A trained writer-verification model to replace the current rule-based
  cosine-similarity proxy for "writer style similarity"
  (`src/evaluation/metrics.py::writer_style_similarity`).
- Run the human-evaluation protocol described in docs/experiments.md with
  actual raters.
- The reference-vs-generated / increasing-reference-data experiments in
  docs/experiments.md are designed but need to be executed and their
  results recorded once a full training run (not the fast dev-config
  training) is available.
