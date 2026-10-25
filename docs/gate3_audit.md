# GATE 3 audit: why VATr's reproduced HWD is 0.6069, not 0.828

Status: **GATE 3 is OPEN. Check 3.2 is NOT PASSED.** Reproduced HWD 0.6069 vs published 0.828 (rel_diff 0.267, gate range
0.745-0.911). The 0.6069 is the value reported from the Colab `results/vatr_reproduction.json` (that file is not on this
host). Nothing below changes a threshold. This document is a reading audit; none of the Colab cells at the end has been run.

## 1. The published protocol (what 0.828 is)

Source: Pippi et al., *HWD*, BMVC 2023, arXiv 2310.20316 (text extracted from the PDF), section 3.1 and Table 2; and the
HWD repo history (`compute_hwd.py` at 6a15ffd).

| item | published protocol |
|---|---|
| backbone | VGG16 pretrained on Font-Square (10400 fonts), last conv block |
| input | "the whole image resized to a height of 32 pixels while keeping the aspect ratio ... we feed them to the network one by one to avoid using padding" |
| representation | feature map 1 x W x 512 -> a SET of W 512-d vectors per image (wide images give more vectors) |
| aggregation | for writer m, "all the vectors from all the real images sets are gathered and then averaged (the same applies to the generated images)"; longer words therefore weigh more |
| distance | Euclidean distance between the two means (per writer m; averaged over writers) |
| Table 2 | "scores of the considered HTG models when generating the same or different words as those in the style reference images of the IAM test set": HWT 1.928, HiGAN+ 1.237, **VATr 0.828** |
| sample count | not stated in the text; the authors' own `compute_hwd.py` (6a15ffd) names the VATr folders `Fake_test` / `Real_test` and annotates them `# 25766` |
| real set / generated set | `Real_test` / `Fake_test` as written by VATr's `generate_fakes.py` (`vatr_same_words`, `vatr_rand_words` variants in the same file) |

Not stated in the paper: batch size used to generate the images, the RNG state of the style draws, whether Table 2 is the
same-words or random-words variant.

## 2. What `scripts/vatr_reproduce.py` does, and the diff

`vatr_reproduce.py` runs VATr's own `generate_fakes.py` (released `vatr.pth`, `IAM-32.pickle`) and then scores
`FolderDataset(Fake_test)` vs `FolderDataset(Real_test)` with the pinned `HWDScore(height=32)`.

| aspect | paper | ours | note |
|---|---|---|---|
| image pairs | 25766 | 25766 (161 writers) | same count |
| backbone, resize to height 32, token-level features | yes | yes (pinned HWD) | matches |
| per-writer mean then average | yes | yes (`EuclideanDistance`) | matches |
| padding | none | `PaddingSquareHeight` (white-pad to a square when width < height, before the resize) | pinned HWD adds this; only images narrower than tall are touched |
| batch size when VATr wrote the PNGs | unknown (script default 8) | 32 (Colab) | see hypothesis H1 |
| train split generation first | yes (RNG state differs) | skipped, test split starts at seed 742 | changes which 15 words style each fake; expected distance unchanged |
| style words | 15 random words of the writer | same | |

## 3. Pinned HWD `eabb5b5` vs the repo history

Clone made full (`git fetch --unshallow`); the pinned commit is the tip, dated 2025-03-07. The paper-era code is
`6a15ffd` (2023-09-08, last commit before the arXiv submission 2023-10-31). Commits that changed the score between them:
`cefa7da` (2024-02-14 backbone/device refactor), `6151146` (2024-11-02 "Created the HWD module"), `fd7ba25` (stream
option), `c45f910` (2024-11-06 "Fixed the HWD score, now we compute the score for each author and then average as described
in the paper"), `6a405c9`, `e3c87b1`. The repo has no release tags for HWD itself (tags are dataset names).

Every difference that could change the score:

| # | difference | 6a15ffd (2023) | eabb5b5 (pinned) | direction |
|---|---|---|---|---|
| D1 | feature pooling | `AdaptiveAvgPool2d((1,1))`: ONE 512-d vector per image | token-level: every column of the last map is a vector | token-level lets long images dominate the mean, as the paper's text describes; the 2023 code does not |
| D2 | aggregation over writers | ONE pooled distance over the whole set (`score(real, fake)`) | per-writer distance, averaged over writers | per-writer means are noisier, which inflates the distance |
| D3 | transform | `ResizeHeight(32)`, `ToTensor`, `PaddingMin(32,32)` (zero/black right padding up to width 32) | `PaddingSquareHeight` (white pad to square) then `ResizeHeight(32)` | affects narrow images only |
| D4 | resize interpolation | NEAREST | NEAREST | same |
| D5 | batching | batch 1 | batch 1 | same |
| D6 | weights | `VGG16_class_10400.pth` | same file | same |

The pinned code follows the paper's text on D1 and D2; the 2023 commit does not. So the pinned version is not obviously the
wrong one, and the 2023 number is a cross-check, not a target.

## 4. Hypotheses (ranked) and what each predicts

* **H1, white padding written by VATr itself.** `TextCollator` allocates `torch.ones` images padded to the batch's widest
  image, and `save_images_for_fid_calculation` writes those padded tensors to `Real_test` (read from the code). The fakes are generated
  for the whole batch at one width (text encodings are padded to the batch's longest word); that their tails are white is
  expected but NOT verified here, the audit's width columns check it. Padding
  columns become VGG tokens in both sets and pull the two means together. A larger batch means more padding, so batch size 32
  should give a lower token-level HWD than batch 8. This matches the direction of the gap (ours is lower). Prediction:
  the `cropwhite` variant of the audit rises toward 0.83; a regeneration at `--batch_size 8` rises too.
* **H2, 2023 vs 2025 HWD code (D1-D3).** If the paper's number was produced by the 2023 code, then `img_pooled` at
  `era2023` should land near 0.828 and the pinned number should not. The audit scores both on the same pairs.
* **H3, style-draw RNG (train split skipped).** Changes which same-writer words style each fake. It cannot shift the
  expected distance; its size is bounded by the bootstrap interval over writers in the audit. Restoring it needs the train
  split generation (hours on GPU) and is not worth it unless H1 and H2 both fail.
* **Scale checks.** The same-writer floor (real half vs disjoint real half) and the different-writer value (writer w vs w+1)
  place 0.6069 on the metric's scale. The floor uses half-size sets per writer, so it is inflated by small-sample noise
  relative to a full-size comparison; read it as an upper bound on the floor.

## 5. Colab cells to run (do not run them here)

All use the 51.6k images already at `data/processed/vatr_repro/vatr/{Fake_test,Real_test}` on Colab and need the HWD repo
(`third_party/HWD`) from notebook cell 10's clone lines. `scripts/vatr_hwd_audit.py` caches VGG features per image once and
derives every score from them; locally it was verified on synthetic pairs (derived vs real library, pinned: 0.9198963 vs
0.9198964; `6a15ffd`: 0.5311552 vs 0.5311552).

**Cell A, one run answers the same-writer floor, the different-writer value, the paper-era commit on a 2000-pair subset, H1
(cropwhite) and the sampling-noise bound:**
```python
!git pull origin master
!git -C third_party/HWD fetch --unshallow --tags
!python scripts/vatr_hwd_audit.py --device cuda --subset 2000 --seed 0 2>&1 | tee logs/vatr_hwd_audit.log | tail -80
```
Where to read each answer in `results/vatr_hwd_audit.json`:

| question | key |
|---|---|
| full-set score per variant and aggregation | `variants.<pinned\|era2023\|cropwhite>.full.<tok_writer\|tok_pooled\|img_writer\|img_pooled>` |
| fixed 2000-pair subset, pinned code and paper-era code | `variants.*.subset` and `library_checks.pinned_HWDScore32_on_subset`, `library_checks.paper_era_6a15ffd.pooled` |
| same-writer floor | `variants.*.same_writer_floor_real_half_vs_real_half` |
| different-writer value | `variants.*.diff_writer_real_vs_next_writer_real`, `...diff_writer_fake_vs_next_writer_real` |
| fake vs the other half of the same writer's reals | `variants.*.fake_half_vs_real_other_half` |
| H1 (padding) | compare `pinned.full.tok_writer` with `cropwhite.full.tok_writer`; widths in `mean_width_fake/real` |
| H3 (sampling noise) | `variants.*.per_writer_tok.boot_ci95` |

The check that the derived numbers equal the real libraries: `library_checks.pinned_HWDScore32_on_subset` must equal
`pinned_derived_tok_writer_on_subset`, and `paper_era_6a15ffd.pooled` must equal `era2023_derived_img_pooled_on_subset`.

**Cell B, only if H1 is supported: regenerate with the batch size the script defaults to and rescore** (long; a GPU
generation of the test split):
```python
!python scripts/vatr_reproduce.py --device cuda --batch_size 8 --out data/processed/vatr_repro_b8 2>&1 | tee logs/vatr_reproduce_b8.log | tail -40
```
Then compare `results/vatr_reproduction.json` (written by this run; copy the earlier one aside first, it is overwritten) with
the batch-32 value 0.6069.

**Decision rule, fixed in advance:** if some variant or the batch-8 rerun lands in 0.745-0.911 under a protocol that the
paper's text supports, report it as the reproduction under that protocol and say plainly that the gate's own definition
(the pinned `HWDScore(32)` on batch-32 images) still gave 0.6069. If none does, 3.2 is BLOCKED with these findings. Whether
to amend the gate is the user's decision, not made here.
