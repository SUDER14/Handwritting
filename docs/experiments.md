# Experiments

## Research question

> How effectively can a model generate unseen handwritten characters in a
> writer's style when provided with only a small number of handwritten
> reference characters?

## Baseline vs. proposed comparison

| | BASELINE | PROPOSED |
|---|---|---|
| Method | Reference-font glyph + CV style transform | Conditional VAE style encoder + decoder |
| Module | `src/generator/baseline_generator.py` | `src/style_encoder/` |
| Learns from data? | No (deterministic transform) | Yes (trained on EMNIST-letters) |
| Style representation | 4 scalar numbers (height, width, slant, stroke width) | 32-d continuous latent vector |
| Expected strength | Always legible, matches coarse proportions | Should better capture per-letter idiosyncrasy, generalizes from a genuinely learned visual-style manifold |
| Expected weakness | Cannot capture idiosyncratic letterforms (e.g. a distinctive loop) | Bounded by EMNIST's 28x28 resolution and lack of writer-ID supervision |

Run both via `scripts/generate_alphabet.py --backend baseline` /
`--backend neural` on the same input sample, then compare with
`src/evaluation/metrics.py`:
- `character_recognition_accuracy(alphabet)` for each backend's output.
- `writer_style_similarity(alphabet, observed_glyphs)` for each.
- `ssim_against_reference(generated, real)` wherever a held-out real
  glyph for that character exists (leave-one-out, below).

## Leave-one-out protocol (uses real ground truth, not just visual inspection)

For a sample containing multiple characters (e.g. "hello" or a longer
phrase), for each observed character `c`:
1. Temporarily withhold `c` from `observed_glyphs` (treat it as unseen).
2. Generate `c` from the remaining observed characters' style.
3. Compare the generated `c` against the real withheld glyph via
   `ssim_against_reference`.
4. Repeat for every observed character; report mean/std SSIM.

This gives an actual quantitative "did the model get closer to the truth"
number, rather than relying on the eye alone — directly addressing
section 29 of the design brief ("do not claim something works without
testing it").

## Reference-quantity scaling experiment

Research question restated for a controlled variable: does style-embedding
quality (and therefore generated-alphabet quality) improve as more
reference handwriting is provided?

Protocol:
1. Collect (or synthesize, for pipeline testing) handwriting samples of
   increasing size from the same writer: 1 word, 3 words, 5 words, a
   short sentence, multiple sentences.
2. For each size, extract `observed_glyphs` (deduplicated by character)
   and compute `encode_writer_style` (neural) or `StyleProfile`
   (baseline).
3. Generate the full alphabet at each size and score with the metrics
   above (recognition accuracy, writer-style similarity, and — for any
   character observed at every size — leave-one-out SSIM).
4. Plot metric vs. number-of-reference-characters.

Expected shape (hypothesis, to be confirmed by running the protocol): a
concave curve — most of the benefit comes from the first ~10-15 distinct
characters (covering most common letters), with diminishing returns
after, since averaging more style latents mainly reduces estimation
variance rather than adding new information once most letters have been
seen at least once.

Status: protocol implemented and runnable end-to-end (all pieces —
`analyze_sample`, `encode_writer_style`, `generate_alphabet_neural`,
`src/evaluation/metrics.py` — exist and are unit-tested), but not yet
executed against multiple real reference-size conditions and recorded
here. See docs/development_log.md for what has actually been run this
session, and docs/future_work.md for this as a concrete next step.

## Human evaluation protocol (designed, not automated)

For a set of generated-alphabet outputs (varying backend / reference
size), recruit raters to score, per generated word/sentence image, on a
1-5 scale:
- **Similarity to reference handwriting**: "Does this look like it was
  written by the same person as the reference sample?"
- **Naturalness**: "Does this look like real human handwriting (vs.
  obviously synthetic/robotic)?"
- **Readability**: "Can you read this text without difficulty?"

Aggregate mean +/- std per condition; a paired comparison (baseline vs.
proposed, same input) is more informative than absolute scores given
likely small rater pools.
