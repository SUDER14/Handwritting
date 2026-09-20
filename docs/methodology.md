# Methodology

## Problem framing

Given a small handwriting sample containing only some characters
(e.g. "quick" -> {q, u, i, c, k}), generate the remaining 21 lowercase
letters in a way that is recognizably the *same person's* handwriting,
then use the resulting personalized alphabet to render arbitrary text.

This is a **few-shot, writer-conditional character generation** problem.
It decomposes into:

1. **Perception**: turn a photo into labeled glyph images (preprocessing,
   segmentation, recognition).
2. **Representation**: turn glyph images into a writer-style summary
   that generalizes beyond the observed characters (feature extraction /
   style encoder).
3. **Generation**: produce plausible glyphs for unobserved characters,
   conditioned on that style summary (baseline transform / conditional VAE).
4. **Composition**: assemble glyphs (real + generated) into renderable text.

## Two parallel generation methods (by design, not accident)

Both are implemented and directly comparable (`src/evaluation/metrics.py`,
`docs/experiments.md`):

- **BASELINE** (`src/generator/baseline_generator.py`): deterministic
  image-processing. A neutral reference-font glyph is scaled to the
  writer's measured height/width, sheared to match measured slant, and
  dilated/eroded to match measured stroke width. No learned
  representation of style — this is the "simple glyph extraction +
  nearest style transformation" comparison point required by section 21
  of the design brief.
- **PROPOSED** (`src/style_encoder/`): a conditional VAE trained on
  EMNIST-letters learns a continuous style latent space from many
  writers. At inference, a new writer's few observed glyphs are encoded
  and averaged into one style vector, which conditions generation of
  every other letter.

The baseline is not a strawman to be discarded — it's the reliability
fallback: it always produces a legible, correctly-shaped alphabet (it's
built from real font outlines) even before/without a trained model, and
it never depends on how well the neural model happened to converge for a
particular writer. The hybrid policy used in the app (section 25: real
observed glyphs preserved verbatim, only unobserved ones generated)
applies to both backends identically.

## Training regime

```
TRAINING (once, offline):
    EMNIST-letters (145,600 images, 26 classes, thousands of distinct
    writers merged into one un-labeled-by-writer pool)
        -> conditional VAE learns:
             - a content embedding per letter class (supervised)
             - a style latent distributed ~N(0, I) that must explain
               everything about the image the class label doesn't
    (scripts/train_style_encoder.py)

INFERENCE (per new user, seconds):
    a handful of glyphs from one new writer, with known labels
        -> encode each with the trained StyleEncoder -> average
        -> one writer_style vector
        -> decode(writer_style, class_embedding[c]) for every c in a-z
    (src/style_encoder/inference.py)
```

No retraining happens per user. This matches section 11 of the design
brief exactly: the expensive step (learning what "handwriting style"
even means, across enough examples to generalize) happens once, offline,
on a multi-writer corpus; the cheap step (adapting to one new writer)
happens at inference time via latent averaging only.

## Why EMNIST (not IAM/CVL) for this stage

IAM and CVL are word/line-level datasets with realistic multi-word
handwriting and (in IAM's case) writer IDs, which would in principle let
us train an actual writer-identification/verification model (the
"ideal" writer-similarity metric from section 20). Both require
registration and (for IAM) an accepted-use agreement, and both are much
larger and slower to preprocess (full-page scans -> line/word
segmentation -> character alignment) than what an MVP on an 8GB/no-GPU
machine can responsibly take on first. EMNIST-letters gives the same
"26-class, many-writer" character-level signal needed to train the style
encoder, with zero preprocessing (already 28x28, already segmented,
already labeled), so it was chosen to keep Stage 2-4 (see
docs/development_log.md) tractable within this session. Upgrading to a
writer-labeled dataset like IAM is documented as a concrete next step in
docs/future_work.md — it does not require changing the model
architecture, only the training data and (for writer verification) an
additional discriminator head.

Note on data source: `torchvision.datasets.EMNIST`'s built-in downloader
points at a NIST-hosted mirror that has served an **expired TLS
certificate** for an extended period (a known upstream issue). Rather
than weakening TLS verification to force that connection through, this
project sources the identical academic dataset from a Hugging Face Hub
mirror (`tanganke/emnist_letters`) over a normally-verified HTTPS
connection — see `src/utils/emnist_source.py` and
docs/limitations.md.

## Evaluation philosophy

Section 20 asks for quantitative evaluation beyond visual inspection.
Three automatable metrics are implemented (`src/evaluation/metrics.py`):

1. SSIM against a real reference glyph (leave-one-out protocol described
   in docs/experiments.md).
2. Character recognition accuracy of generated glyphs, via the
   non-learned template-matching recognizer used as an OCR proxy.
3. Writer-style similarity: cosine similarity between rule-based
   `StyleProfile` feature vectors of the generated alphabet vs. the
   original sample — a proxy for writer-identity preservation, pending a
   real writer-verification model (future work).

Human evaluation (naturalness/similarity ratings) is designed as a
protocol in docs/experiments.md but is not automated — it requires human
raters outside the scope of an automated pipeline.
