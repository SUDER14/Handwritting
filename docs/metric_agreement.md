# Writer-ID distance vs HWD (ROADMAP Task 5.3)

**Question.** In the protocol table the glyph VAE is *closer* to the real writers than the font baseline by writer-ID
distance (K=1: 0.4528 vs 0.6861) but *farther* by HWD (4.354 vs 3.443). Do the two style metrics measure different things,
and what? Every number below is from `results/metric_agreement/{samples.csv,summary.json}`
(`scripts/metric_agreement.py` -> `scripts/analyze_metric_agreement.py`; figures `results/figures/metric_scatter.png`,
`metric_disagreement.png`). Same 20 test writers (seed 1337), K=1, same pinned instruments, CPU-reduced, **per sample**:
200 real, 200 baseline, 200 glyph_vae lines (none failed). `wid` = 1 - cosine(embedding of the line, mean embedding of the
writer's real held-out lines); `hwd` = official HWD between the line and the writer's real held-out lines. Real lines are
scored leave-one-out.

## 1. The pattern replicates per sample

| | wid (mean) | HWD (mean) | protocol table, K=1 (set level) |
|---|---|---|---|
| real | 0.0520 | 1.394 | 0.0158 / 0.771 (half vs half) |
| baseline | 0.6887 | 3.542 | 0.6861 / 3.443 |
| glyph_vae | 0.4592 | 4.481 | 0.4528 / 4.354 |

glyph_vae is closer on writer-ID (0.459 vs 0.689) and farther on HWD (4.48 vs 3.54) here too.

## 2. How much do the two metrics agree?

Spearman rho between wid and HWD over samples:

| subset | rho | n | p |
|---|---|---|---|
| all (real + baseline + glyph_vae) | 0.5767 | 600 | 1.8e-54 |
| real | 0.4706 | 200 | 2.0e-12 |
| baseline | 0.2009 | 200 | 0.0043 |
| glyph_vae | 0.1110 | 200 | 0.118 |

Within-writer (mean of the 20 per-writer rhos, removes writer differences): real 0.3442, baseline -0.0558, glyph_vae
0.1273. So the overall 0.58 is mostly the gap between real and synthetic lines (both metrics call real lines close and
synthetic lines far). **Among generated lines the two metrics are nearly uncorrelated.** The overall rho is above the
ROADMAP's 0.5 trigger, but the per-backend values are not, so the 10 most-disagreeing samples are saved anyway
(`results/metric_agreement/disagreement_examples.csv`, `results/figures/metric_disagreement.png`).

The 10 examples (rank-percentile difference over all 600 samples): five baseline lines of writer 513 (writer-ID percentile
0.95-0.99, HWD 0.39-0.46: "writer-ID far, HWD near") and five glyph_vae lines of writers 173, 557, 547 (writer-ID percentile
0.35-0.44, HWD 0.98-0.997: "writer-ID near, HWD far"). Looking at them: the glyph_vae lines are heavy black blobs with a few
real-looking crops; the baseline lines are thin, light-weight text partly built from real crops of the writer's own
reference (it reuses the writer's seen letters), against a pencil-grey reference.

## 3. Hypotheses that were tested, and what the data says

Hypothesis (stated before looking): *the writer-ID net responds to ink weight / texture statistics, HWD responds to stroke
geometry (letterform shape, lean).* Two kinds of test.

**(a) Descriptors of each line vs the writer's real lines** (native ink masks; slant, stroke width, ink density,
components per word, measured as in `src/analysis/writer_features.py`):

| mean over samples | stroke width (px) | ink density | lean (deg) | components per word |
|---|---|---|---|---|
| real | 5.18 | 0.0953 | 10.2 | 4.29 |
| baseline | 2.84 | 0.0231 | 2.2 | 7.68 |
| glyph_vae | 16.50 | 0.0768 | 3.0 | 6.97 |

Relative gap to the writer's own real lines (mean): ink density baseline 0.747, glyph_vae 0.310; stroke width baseline
0.403, glyph_vae 2.329; lean (deg) baseline 8.99, glyph_vae 13.97; components per word baseline 1.02, glyph_vae 0.74.
So glyph_vae is closer on ink mass (consistent with the writer-ID result) and much farther on stroke width and lean.

Spearman of each metric with these gaps. Pooled over the 400 generated lines the signs are confounded by the backend
(glyph_vae has a small density gap and a huge stroke gap), so the within-backend values are the ones to read:

| within backend | wid vs stroke gap | wid vs density gap | HWD vs stroke gap | HWD vs density gap | HWD vs lean gap |
|---|---|---|---|---|---|
| baseline | 0.661 | 0.475 | 0.436 | 0.559 | 0.149 |
| glyph_vae | -0.084 (ns) | 0.037 (ns) | 0.392 | -0.145 | 0.222 |

Within baseline both metrics follow the weight gaps; within glyph_vae writer-ID follows neither weight gap while HWD follows
the stroke-width and lean gaps. This is correlational and mixed; it does not by itself separate the two metrics.

**(b) Interventions** (the cleaner test): repair ONE aspect of each generated line, keep the rest, rescore. `stroke`
= dilate/erode to the writer's real median stroke width; `slant` = shear to the writer's real median lean; change = after -
before; "improved" = share of lines whose distance went down.

| backend | edit | change in wid | change in HWD |
|---|---|---|---|
| baseline | stroke (gap 2.27 -> 0.22 px) | **-0.2135 (-31.0%)**, improved in 76% | +0.0234 (+0.7%), improved in 30% |
| baseline | slant (gap 8.99 -> 2.72 deg) | +0.0008 (+0.1%) | -0.0373 (-1.1%) |
| baseline | both | -0.2238 (-32.5%) | -0.0791 (-2.2%) |
| glyph_vae | stroke (gap 11.39 -> 7.77 px, only partly achievable) | +0.1018 (+22.2%), improved in 22% | +0.0498 (+1.1%) |
| glyph_vae | slant (gap 13.97 -> 1.13 deg) | -0.0718 (-15.6%), improved in 73% | **-0.3539 (-7.9%)**, improved in 70% |
| glyph_vae | both | +0.0700 (+15.2%) | -0.2023 (-4.5%) |

## 4. What can be concluded, and what cannot

Supported by the data:

1. **The writer-ID net is sensitive to ink weight, and (in the baseline) not to lean.** Thickening the baseline's thin
   strokes to the writer's width cuts its writer-ID distance by 31% while HWD does not move (+0.7%); shearing it changes
   writer-ID by 0.1%. (Not blind to lean in general: shearing glyph_vae lowers its writer-ID distance by 15.6%.) Making glyph_vae's already too-thick strokes thinner makes writer-ID *worse* (+22%), which fits a
   response to overall ink mass/density rather than to stroke width as such (glyph_vae's ink density, 0.077, is close to
   real 0.095; the baseline's, 0.023, is a quarter of it). I did not measure ink density after the edits, so "mass vs
   width" is inferred, not isolated.
2. **HWD does not respond to ink weight in these lines** (|change| <= 1.1% for both stroke edits) but does respond to lean:
   shearing glyph_vae to the writer's lean lowers its HWD by 7.9%.
3. Together this accounts for the sign flip qualitatively: glyph_vae's heavy, dense ink looks closer to the writers to the
   writer-ID net, and HWD ignores that advantage.

Not established:

* **Most of the HWD gap is not explained.** Repairing lean closes only 0.354 of the 0.939 HWD gap between glyph_vae (4.481)
  and baseline (3.542), i.e. about 38%. The rest is something these edits did not touch. The montage suggests letterform
  shape (blob glyphs vs real crops/font glyphs), consistent with HWD's backbone being trained to classify fonts, but that is
  a suggestion from 10 pictures, not a test.
* The glyph_vae stroke edit only reduced the stroke gap from 11.39 to 7.77 px (8 erosion steps), so "HWD ignores stroke width
  for glyph_vae" is a statement about a partial repair.
* Per-sample numbers from 20 writers and one K; the writer-ID net and HWD are the CPU-reduced / pinned instruments, and the
  conclusions concern these two instruments, not metric families in general.
* Rho between two metrics says nothing about which is right; neither was validated here against human judgement (task 5.4
  is waiting for the user's ratings).
