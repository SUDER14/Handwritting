# Architecture

## Pipeline overview

```
Handwriting photo (PNG/JPG)
        |
        v
[1] Preprocessing   (src/preprocessing/pipeline.py)
    grayscale -> denoise -> CLAHE contrast -> adaptive+Otsu binarize
    -> deskew (minAreaRect) -> crop to ink bounding box
        |
        v
[2] Segmentation    (src/segmentation/segmenter.py)
    connected components -> diacritic merge (dot-on-i/j) -> left-to-right
    order -> reconciled to len(label_text) via merge/split
        |
        v
[3] Recognition     (src/recognition/*)
    pluggable: labeled (ground truth) | template-matching | cnn (trained classifier)
        |
        v
[4] Feature extraction (src/features/style_extractor.py)
    per-glyph: height, width, slant (PCA), stroke width (distance
    transform), curvature (isoperimetric ratio), ink density
    writer-level: means/std aggregated into StyleProfile
        |
        +----------------------------+
        |                            |
        v                            v
[5a] BASELINE generator        [5b] PROPOSED generator
   (src/generator/                (src/style_encoder/*)
    baseline_generator.py)         CNN StyleEncoder -> style latent (mu)
   reference-font glyph +          averaged over observed glyphs
   CV style-transform              -> Decoder(style, char_embedding)
   (scale/shear/morphology)        -> generated glyph
        |                            |
        +----------------------------+
                     |
                     v
        Hybrid alphabet: real observed glyphs preserved,
        unobserved glyphs generated (either backend)
                     |
                     v
[6] Alphabet export  (src/generator/alphabet_builder.py)
    outputs/alphabets/<session>/lowercase/*.png + glyph_metadata.json
                     |
                     v
[7] Text renderer    (src/renderer/text_renderer.py)
    per-occurrence variant selection + baseline/rotation jitter
    -> arbitrary text as a handwriting-style image
```

## Why a conditional VAE (not GAN / diffusion)

Section 24 of the design brief asks for the simplest architecture that
demonstrates the research idea, with rationale documented. Given the
target machine (Windows, 4 CPU cores, 8GB RAM, **no GPU** — confirmed via
`torch.cuda.is_available() == False` during environment inspection):

| Architecture | Training stability | Compute for MVP-quality result | Fits this machine |
|---|---|---|---|
| GAN | Notoriously unstable (mode collapse, needs careful tuning) | High — typically needs many more epochs/tricks to converge | Risky |
| Diffusion | Stable but sample generation requires many forward passes | Very high — impractical to train from scratch on CPU in a dev session | No |
| **Conditional VAE** | Stable, single well-defined loss | Low — converges in a handful of epochs on 28x28 grayscale glyphs | **Yes** |
| Siamese / plain autoencoder | Stable, simple | Low, but no generative decoder — can't synthesize unseen glyphs | Insufficient (no generation) |

The VAE is also the architecturally right fit for the **few-shot
averaging** step central to this project: the latent space is smooth and
Euclidean, so averaging several observed glyphs' encoded `mu` vectors
into one writer-style embedding is mathematically well-motivated (it's
literally averaging Gaussian posterior means). A GAN's noise-to-image
mapping has no such natural "average several real examples into one
style code" operation without extra machinery (e.g. a separately trained
encoder, which is closer to what a VAE already gives for free).

## Content/style disentanglement mechanism

```
IMAGE = CHARACTER CONTENT + WRITER STYLE
```

Enforced architecturally, not just hoped for:

- `StyleEncoder` (a CNN) sees **only the pixel image** — it never receives
  the character label as input. Its output (`mu`, `logvar`) is the sole
  channel through which image information reaches the decoder's "style"
  input.
- `CharacterEmbedding` is a learned lookup table, one vector per class
  (a-z), supplied to the decoder **independently** of the encoder.
- `Decoder` reconstructs the image from `[style_latent ; content_embedding]`.

Because the decoder needs *some* signal to know which of the 26 letters
to draw, and the only per-class signal available is `content_embedding`
(not anything derived from the encoder), the encoder has no incentive to
spend latent capacity encoding "which letter is this" — that information
is already provided for free via `content_embedding`, supervised directly
by the training label. What's left for the KL-regularized style latent to
usefully encode is everything else that affects the reconstruction:
slant, stroke width, roundness, proportions — i.e., style.

## Hardware-driven choices (section 23)

Environment inspection results (recorded at project start):

- OS: Windows 10 Pro, 4 CPU cores, 8.4 GB RAM
- Python 3.12.3
- PyTorch 2.14.0 **CPU build** (`torch.cuda.is_available() == False`)
- No discrete GPU / CUDA toolkit present

Consequences for design:
- Model kept small (three conv layers each in encoder/decoder, 28x28
  inputs) so a full epoch over ~38k images completes in low
  single-digit minutes on CPU.
- `training.max_train_samples` in `config/config.yaml` caps the EMNIST
  subset used per run (default 40,000 of the available ~124,800) to keep
  iteration fast during development; raise it for a longer/better final
  training run once the pipeline is validated.
- Batch size 128, 8 epochs by default — tuned empirically for a CPU
  budget of a few minutes' wall time (see docs/development_log.md for
  actual observed timings).

## Module map

| Module | Responsibility | Replaceable via |
|---|---|---|
| `src/preprocessing/pipeline.py` | Raw image -> clean binary line image | N/A (shared foundation) |
| `src/segmentation/segmenter.py` | Line image -> ordered glyph crops | `segmentation.method` config key (future: projection-profile backend) |
| `src/recognition/*` | Glyph crop -> character label | `recognition.backend` config key (`labeled` / `template` / `cnn`) |
| `src/features/style_extractor.py` | Glyphs -> interpretable style numbers | N/A (used for UI + baseline generator + evaluation) |
| `src/generator/baseline_generator.py` | Style numbers -> synthesized glyphs (non-learned) | Swap for any other CV heuristic |
| `src/style_encoder/model.py` + `inference.py` | Learned style embedding + conditional generation | Retrain with different architecture, same interface |
| `src/renderer/text_renderer.py` | Alphabet -> arbitrary rendered text | N/A |
| `src/evaluation/metrics.py` | Quantitative comparison of the two generator backends | N/A |
