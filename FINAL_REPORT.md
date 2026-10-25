# FINAL REPORT (draft, written with GATE 3 OPEN)

> **This report is complete except for the VATr numbers, which are marked `PENDING COLAB` and are not estimated anywhere.**
> **GATE 3 is OPEN and not passed:** the VATr reproduction gave HWD **0.6069** against the published **0.828**
> (relative difference 0.267; the gate requires 0.745-0.911). Stages 4-6 were worked past the open gate at the user's explicit,
> session-only override (logged in `ROADMAP_STATE.md`); no threshold was changed and GATE 3 is not logged as passed anywhere.
> Every number below was read from a file in `results/`, `ROADMAP_STATE.md`, a log, or a command run in this work; the source is named.

## 1. What was built

A study of few-shot, line-level handwriting synthesis on unseen IAM writers. Three generators (font-glyph **baseline**,
EMNIST **glyph VAE** as an ablation, **VATr** through its released checkpoint) are run under one protocol: K reference lines
-> generate the writer's other lines -> score with frozen instruments (CRNN CER for legibility; a from-scratch writer-ID net
and the official HWD for style fidelity). Plus: a user-sample pipeline (line -> words -> K references), per-writer and
metric-agreement analyses, a human-check set, a Streamlit app, and the VATr reproduction/audit tooling.

## 2. Labels that apply to every number

* **CPU-reduced**: no CUDA on the host, so all trained-model caps are divided by 4 (writer-ID net 10 epochs = 40/4; CRNN 15 epochs = 60/4;
  EMNIST VAE 6000 images x 6 epochs instead of 40000 x 8).
* **20-writer subset**: every protocol row scores the same seeded 20 of the 161 IAM test writers (seed 1337), 10 target
  lines each (fewer at large K, see section 4).
* **GPU**: nothing in this report was produced on a GPU except the user-reported Colab VATr reproduction (0.6069).
* Instruments are the pinned GATE-1 runs `writerid_20260921T180911Z_ed897d02_cpu_r1` and `crnn_20260922T003942Z_d1f87b49_cpu_r1`.

## 3. Results table (`results/protocol_table.csv`)

Lower is better in every column.

| backend | K | CER | writer-ID distance | HWD | label |
|---|---|---|---|---|---|
| real (upper bound; half the held-out lines vs the other half) | 1 | 0.1038 | 0.0158 | 0.771 | 20-writer subset, CPU-reduced instruments |
| real | 5 | 0.0934 | 0.0215 | 0.898 | same |
| baseline | 1 | 0.8001 | 0.6861 | 3.443 | same |
| baseline | 5 | 0.8352 | 0.6765 | 3.463 | same |
| glyph_vae | 1 | 0.7954 | 0.4528 | 4.354 | same; VAE CPU-reduced |
| glyph_vae | 5 | 0.8086 | 0.4994 | 4.244 | same |
| **VATr** | 1 | **PENDING COLAB** | **PENDING COLAB** | **PENDING COLAB** | Colab GPU, pinned instruments |
| **VATr** | 5 | **PENDING COLAB** | **PENDING COLAB** | **PENDING COLAB** | same |

Out-of-vocabulary subset (lines containing a letter absent from the K references) is in the table's `oov_*` columns; at K=1
every writer has one, so the K=1 OOV row equals the overall row. Both synthetic backends are close to illegible to the frozen
recognizer (CER about 0.8; the recognizer reads real lines at 0.10).

## 4. K-sweep (Task 5.1; baseline only, VATr PENDING COLAB)

Same 20 writers, same pinned instruments (`results/figures/k_sweep.png`, `k_sweep_baseline.csv`; K=1 and K=5 come from the
earlier runs, K=3/10/15 from the new run). Writers need K+1 eligible lines and at least 2 target lines, so writers drop out
at large K; they are listed, not replaced:

| K | writers scored | targets scored (of 200) | CER | writer-ID | HWD | not scored |
|---|---|---|---|---|---|---|
| 1 | 20 | 200 | 0.8001 | 0.6861 | 3.4427 | |
| 3 | 20 | 195 | 0.8213 | 0.6884 | 3.4529 | |
| 5 | 20 | 183 | 0.8352 | 0.6765 | 3.4630 | |
| 10 | 18 | 131 | 0.8035 | 0.7030 | 3.5632 | 196, 594 |
| 15 | 11 | 78 | 0.9135 | 0.7053 | 3.4964 | 172, 179, 190, 196, 283, 553, 557, 561, 594 |

Because the writer set shrinks, the like-for-like trend uses the 11 writers scored at every K: CER 0.8274 / 0.8686 / 0.8747 /
0.8575 / 0.9135 and HWD 3.3593 / 3.3821 / 3.4438 / 3.4478 / 3.4964 for K = 1 / 3 / 5 / 10 / 15. **More reference lines did not
help the baseline** (HWD rises about 4% from K=1 to K=15 on the common writers; CER does not improve). VATr curve: PENDING COLAB.

## 5. Writer-ID distance vs HWD (Task 5.3; `docs/metric_agreement.md`)

The glyph VAE is closer than the baseline by writer-ID distance (0.4528 vs 0.6861 at K=1) but farther by HWD (4.354 vs 3.443).
Per sample (200 lines per backend): Spearman rho between the two metrics is 0.5767 overall (n=600) but only 0.2009 for the baseline
and 0.1110 for glyph_vae (p 0.118): among generated lines the two metrics are nearly uncorrelated; the overall value mostly reflects
real-vs-synthetic. Tested explanation (interventions that repair one property of each generated line):
shaping the baseline's stroke width to the writer's lowers its writer-ID distance by 31.0% and leaves HWD within +0.7%;
thinning the glyph VAE's too-thick strokes raises its writer-ID distance by 22.2% (HWD +1.1%); shearing the glyph VAE to the writer's
lean lowers HWD by 7.9% (and writer-ID by 15.6%). So the writer-ID net follows ink weight/density (glyph_vae's density gap to the writers
is 0.31 vs the baseline's 0.75) and HWD ignores it. **Only about 38% of the HWD gap between the two backends is explained** by lean; the rest is
untested (the example montage suggests letterform shape). Ten most-disagreeing samples: `results/figures/metric_disagreement.png`.

## 6. Per-writer analysis (Task 5.2; `results/figures/per_writer_*`)

Correlating each writer's CER/HWD with slant, stroke width and cursiveness measured on their real test lines (n=20 writers; 18 / 11 at K=10 / 15):
42 CER/HWD cells, 4 with p<0.05, all baseline CER (slant at K=3 rho +0.78 p=0.0001, K=10 +0.64, K=15 +0.65; cursiveness at K=3 -0.67). The same slant-CER
cell is +0.04 at K=1 and +0.29 (p 0.22) at K=5, so **nothing here is stable across K**; 4 of 42 is about chance. No cell is significant for glyph_vae.
Exploratory only.

## 7. GATE 3 status

| check | state |
|---|---|
| 3.2 reproduced VATr HWD within 10% of 0.828 | **NOT PASSED**: 0.6069 (rel. diff 0.267, range 0.745-0.911; 25766 fake/real pairs, 161 writers, pinned HWD, `--batch_size 32`; value reported by the user from the Colab file) |
| 3.5 VATr rows K=1, K=5 | PENDING COLAB (cell 10b; `scripts/merge_vatr_rows.py` will validate and merge) |
| 3.6 samples for all backends | real, baseline, glyph_vae present (`results/*_samples_*`); VATr PENDING COLAB (cell 10c) |
| leakage assertion | holds for the trained instruments (GATE 1 log) |

The reproduction came out *better* than published, which suggests a protocol difference rather than a generation bug.
`docs/gate3_audit.md` has the diff against the paper's protocol (per-writer mean of token features at height 32, no padding),
the differences between HWD `eabb5b5` (pinned) and the Sept-2023 code `6a15ffd` (global pooling, pooled distance), the leading
untested hypothesis (VATr writes batch-width white padding into both image folders; the Colab run used batch 32), and the exact
Colab cells (`scripts/vatr_hwd_audit.py`, plus a batch-8 regeneration). **Not run.** Whether to amend the gate is the user's decision.

## 8. Earlier gates (from `ROADMAP_STATE.md`)

* GATE 0 PASSED: join match rate 0.9024 (9360/10373), 0 conflicting form writers, 0 train/test writer overlap.
* GATE 1 PASSED: writer-ID test top-1 0.7319 (mAP 0.5650; chance 0.011); HWD same < different in 200/200 pairs; CRNN test CER 0.1023 (WER 0.3272, 2666 lines); real row present.
* GATE 2 PASSED: six rows (real, baseline, glyph_vae x K=1,5), shear test passes.
* GATE 4 PASSED: word count within +-1 on 0.865 of 200 test lines (first rule 0.650 failed, the fix 0.865); deskew |angle| < 2 deg on synthetic text (and p95 1.0 deg on real lines); 4.3 DEFERRED.

## 9. Deviations and incidents (all logged when they happened)

1. **Writer-ID net is not a standard ResNet-18**: block layout [2,2,2,2] but widths 32/64/128/256 (2.87M parameters), kept because existing code/tests used it.
2. **ImageNet `resnet18` downloaded during the first VATr smoke test** (45 MB, `download.pytorch.org`, not on the allow-list): VATr's constructor fetched it implicitly; the checkpoint overwrites all its tensors so no result depends on it; a runtime shim now builds it with `weights=None`.
3. **`forms.txt` is in public git history** (committed in `5150ef8`, untracked and gitignored since `5457f3b`, history not rewritten by the user's choice). It is IAM's licensed file.
4. **CPU-reduced training caps** (section 2) and a 20-writer evaluation subset, because there is no CUDA.
5. **Join is text-only**: Teklia/IAM-line has no line id or writer; 1013 of 10373 rows are excluded (734 ambiguous, 276 unmatched, 3 outside the split), never guessed.
6. **3.4 VATr fine-tuning skipped** (needs >= 12 GB VRAM; the released checkpoint is evaluated as is).
7. **Stages 4-6 ran ahead of GATE 3** at the user's explicit request (Stage 4 earlier, 5-6 in this session).
8. **Harness low-memory reaper** killed several runs (an evaluation run, the Stage-4 measurement, and in this session a test run containing the end-to-end app test and two waiting shells). Detached jobs were not affected; no result comes from a reaped run.
9. K=1 and K=5 baseline rows came from code `d13db30`; the K=3/10/15 rows from `0448e6b`, whose only change is the K-sweep bookkeeping (writer subset fixed by the configured ks; per-K target coverage recorded).
10. The VATr protocol rows, the HWD audit and the batch-8 regeneration were not run at all here (no GPU); see section 10.

## 10. DEFERRED / pending items

| item | what is needed |
|---|---|
| **Colab VATr rows (3.5)** and **VATr samples (3.6)** | run cells 10b/10c with the two pinned `best.pt` uploaded (the CRNN `best.pt` was missing on Drive on 2026-10-20), then `scripts/merge_vatr_rows.py` |
| **3.2 diagnosis** | run `scripts/vatr_hwd_audit.py --device cuda` on Colab, send `results/vatr_hwd_audit.json` |
| **4.3 real photos** | `data/user_photos/` ([HUMAN]) |
| **5.4 human ratings** | rate `results/human_check/human_check.csv` ([HUMAN]); VATr pairs added after Colab |
| VATr points on the K-sweep, per-writer analysis, metric-agreement analysis | after the Colab rows are merged (`plot_k_sweep.py` and `analyze_per_writer.py` pick up `vatr` rows automatically) |

## 11. What worked, what did not

* Worked: the instruments (writer-ID net 0.7319 top-1, CRNN CER 0.1023, HWD separating same/different writers 200/200), the protocol and its bookkeeping,
  the Teklia join, the word-segmentation rule, the by-name VATr checkpoint load, the merge validator and the interventions analysis.
* Did not: the synthetic backends are nearly unreadable (CER about 0.8) at every K, more references do not help the baseline, the two style metrics disagree about
  which synthetic backend is closer, and the VATr reproduction missed the gate (0.6069 vs 0.828).
* Not claimed: anything about VATr's protocol numbers, any GPU result, or that either style metric matches human judgement.

## 12. Tests

Full suite (`pytest tests`, run detached, `logs/full_suite.out`): **185 passed, 1 skipped, 0 failed** in 227.64 s. The one skip is the `accelerate`/`transformers` import test (`tests/test_wandb_stub.py:36`: `accelerate` is not installed on this host; it can only run on Colab). The suite includes the headless Streamlit smoke tests (launch, VATr switch disabled with a reason, upload-without-file warning) and the end-to-end demo generation with scoring; that end-to-end test had been killed once by the harness low-memory reaper earlier in the session (together with the rest of that run) and passed when the suite was run detached.
