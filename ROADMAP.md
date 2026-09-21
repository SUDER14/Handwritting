# ROADMAP — Handwriting Style Synthesis (autonomous execution)

Goal: given a sample of a person's handwriting, generate arbitrary text,
including cursive and characters never seen in the sample, in that person's
style. Evaluate with a legibility score and a style-fidelity score on real
writers never seen in training.

---------------------------------------------------------------------------
## HOW TO RUN THIS FILE

You (Claude Code) execute this roadmap autonomously, stage by stage, without
waiting for approval between stages, EXCEPT where a gate fails or an item is
marked [HUMAN].

### Resumable state
- Maintain ROADMAP_STATE.md at the repo root. Create it if missing.
- At the start of every session: read ROADMAP_STATE.md, then continue from
  the first task not marked DONE. Never redo a DONE task unless its outputs
  are missing from disk.
- After every task: append to ROADMAP_STATE.md
    `<task id> | DONE / FAILED / BLOCKED / DEFERRED | commit SHA | key numbers | notes`
- Before your context gets long, write a short "RESUME HERE" note at the
  bottom of ROADMAP_STATE.md with exactly what to do next.

### Automated gates
- Each stage ends with a GATE of pass/fail checks with fixed thresholds.
- If all checks pass: write the gate report into ROADMAP_STATE.md and
  proceed to the next stage.
- If a check fails: diagnose and attempt a fix, at most 2 attempts, staying
  within that stage's scope. If it still fails: mark the stage BLOCKED in
  ROADMAP_STATE.md with the numbers, the suspected cause and what you tried,
  then STOP the whole run. Do not continue to later stages.
- NEVER change a gate threshold, loosen a test tolerance, or redefine a
  metric to make a check pass. That counts as a failed gate.

### Long-running jobs
- Launch every training run as a background process that logs to
  logs/<run_id>.log, and poll the log. Do not block the session on training.
- Use early stopping on the validation metric. Hard epoch caps are given
  per task; do not exceed them.
- If CUDA is not available, say so in ROADMAP_STATE.md, reduce every
  training cap by 4x, and continue. Record this so results are labelled.

### Standing rules
- One commit per task. Commit message: what changed + key number.
  Never run an experiment on uncommitted code; metrics.json carries the SHA.
- Checkpoints: models/checkpoints/<run_id>/best.pt and last.pt. Never
  overwrite another run's checkpoint.
- Test-split writers are never seen by any trained component (generator,
  writer-ID network, recognizer). Enforce with an assertion in the data
  loader that fails if a test writer_id appears in a training batch.
- Recognition models are measurement instruments only. Do not work on
  improving recognition beyond what a task specifies.
- Allowed downloads: pip packages, HF datasets Teklia/IAM-line and
  xReniar/IAM-Dataset, the official VATr repo and released checkpoint, the
  official HWD repo and pretrained weights. Anything else, or any single
  download over 5 GB: mark [HUMAN] and defer, do not download.
- Do not install system software. Do not modify files outside this repo.
- Use venv\Scripts\python for everything (system python lacks cv2).
- If this file conflicts with an existing test or doc, this file wins;
  note the conflict in ROADMAP_STATE.md.

---------------------------------------------------------------------------
## STAGE 0 — Close out current work

0.1 Report the Teklia join summary into ROADMAP_STATE.md: duplicate form_ids
    (and whether their writer IDs differ), duplicate line_ids, match rate,
    excluded count by reason, writer and line counts per split.
0.2 Check whether Teklia's own train/val/test split is writer-disjoint;
    record overlapping writer counts per split pair.
0.3 Canonical split for generation = the IAM writer split used by the
    HWT/VATr code (make_vatr_split.py). Seeded split moves to
    data/splits/seeded/ and is marked fallback-only in config.yaml.
0.4 Commit A: join + split code, split files, config, conftest.py.
    Commit B: docs only.
0.5 Fix test_augmentation_adds_no_slant_and_no_stroke_width_change. Record
    which transform broke it and the measured slant / stroke-width deltas.
    Fix the augmentation, not the test. Any writer-ID checkpoint trained
    with the broken augmentation is marked INVALID in ROADMAP_STATE.md.
0.6 Delete AGENTS.md (it belongs to another project). For each config key
    read nowhere: wire it up or delete it; record which.

GATE 0 (all must pass):
- join match rate >= 0.90
- zero form_ids with conflicting writer IDs
- zero writer overlap between train and test in the canonical split
- full test suite passes

---------------------------------------------------------------------------
## STAGE 1 — Measurement instruments

1.1 Writer-ID network (our style-fidelity metric). Retrain after the 0.5
    fix. ResNet-18 from scratch, grayscale input, 64 px height, random
    width window, 256-d embedding, loss = CE + NT-Xent (weight in config).
    Augment only with small affine jitter and elastic distortion; never
    slant or stroke width. Train-split writers only. Cap: 40 epochs.
    Evaluate on test writers: top-1 retrieval and mAP with cosine distance,
    excluding same-form lines from the gallery.
1.2 HWD (literature-comparable style metric, Pippi et al., BMVC 2023).
    Integrate the official implementation and pretrained backbone. Do not
    retrain. Sanity check: sample 200 pairs of test writers (A, B); for each
    compute HWD(A lines vs disjoint A lines) and HWD(A vs B).
1.3 Legibility scorer. Train a CRNN-CTC line recognizer on train-split
    writers only: IAM lines, height 64, conv stack + 2-layer BiLSTM, CTC
    loss, greedy decoding, AdamW. Cap: 60 epochs. Freeze after training.
    Record CER and WER on real test-split lines.
1.4 Evaluation protocol in scripts/evaluate.py. For each test writer:
    take K reference lines (K in {1, 5}, fixed seed), generate the text of
    that writer's OTHER lines, compare against that writer's real held-out
    lines. Per writer, then averaged:
      - legibility: frozen-recognizer CER on generated lines
      - fidelity: writer-ID cosine distance and HWD
      - OOV subset: lines containing characters absent from the K refs
    Also score REAL held-out lines through the same pipeline as the
    "real" upper-bound row. Write everything to results/.

GATE 1 (all must pass):
- writer-ID top-1 retrieval on test writers >= 0.60
- HWD same-writer < HWD different-writer in >= 90% of sampled pairs
- recognizer CER on real test lines <= 0.15
- "real" row exists in results/ with all three metrics
- test-writer leakage assertion passes

---------------------------------------------------------------------------
## STAGE 2 — Baseline on real data

2.1 Fix the baseline shear: shear by (writer_slant - font_slant), where
    font_slant is measured once by our own slant estimator on the rendered
    reference font and cached. Add the test: shear a vertical bar by +20
    degrees, estimator returns +20 +/- 1.
2.2 Run the baseline through the 1.4 protocol, K = 1 and K = 5.
2.3 Run the EMNIST glyph VAE through the same protocol. It is an ablation
    row only. Do no further development on it.
2.4 Record per-character coverage: how often the baseline falls back to the
    font because a character is absent from the references.

GATE 2 (all must pass):
- results table rows present: real, baseline, glyph-VAE, for K=1 and K=5,
  each with CER, writer-ID distance, HWD, and OOV-subset values
- shear test passes
- full test suite passes

---------------------------------------------------------------------------
## STAGE 3 — Line-level generator (main contribution)

3.1 Record GPU name, VRAM, and CUDA availability in ROADMAP_STATE.md.
3.2 Clone the official VATr repository into third_party/VATr. Reproduce its
    published IAM results with the released checkpoint, using its own
    evaluation code, before changing anything.
3.3 Wrap VATr as a third backend behind the same interface as baseline and
    neural. Input: K reference crops. Output: generated line image.
3.4 If VRAM >= 12 GB: fine-tune on the canonical split's train writers.
    Cap: 50 epochs. Otherwise evaluate the released checkpoint only and
    record that fine-tuning was skipped and why.
    If CUDA is unavailable: evaluate on a fixed subset of 20 test writers
    and label all Stage 3 results "subset".
3.5 Run the 1.4 protocol, K = 1 and K = 5. Report the OOV subset separately.
3.6 Save 10 visual samples per backend for the same 3 test writers and the
    same text to results/<run_id>/samples/.

GATE 3 (all must pass):
- reproduced VATr HWD within 10% of the value published in its paper/repo
- VATr rows present in the results table for K=1 and K=5
- sample images present for all backends
- test-writer leakage assertion passes

---------------------------------------------------------------------------
## STAGE 4 — From a user's sample to references

4.1 User-sample pipeline: line image -> word segmentation (vertical
    projection profile with gap-width threshold learned from train-split
    statistics) -> height normalisation -> K references for the generator.
    The neural path no longer requires isolating individual characters.
4.2 Deskew: keep the short-sample skip rule. Measure the skew estimator's
    output distribution on 200 real IAM lines and record it.
4.3 [HUMAN] Real photos of handwriting not from IAM. If data/user_photos/
    contains images, run the pipeline on them and record failures.
    Otherwise mark DEFERRED and continue.

GATE 4 (all must pass):
- on 200 real test-split IAM lines, the predicted word count is within +/-1
  of the transcription's word count for >= 80% of lines
- deskew returns |angle| < 2 degrees on synthetic horizontal text
- full test suite passes

---------------------------------------------------------------------------
## STAGE 5 — Experiments for the write-up

5.1 Legibility vs fidelity tradeoff: for VATr and baseline, vary K in
    {1, 3, 5, 10, 15}; plot CER (x) vs HWD (y); save to results/figures/.
5.2 Per-writer analysis: per test writer, correlate VATr's CER and HWD with
    that writer's slant, stroke width, and cursiveness (mean connected
    components per word on their real lines). Save table and scatter plots.
5.3 Metric agreement: Spearman correlation between our writer-ID distance
    and HWD over all generated samples. If rho < 0.5, save 10 examples where
    they disagree most.
5.4 [HUMAN] Build results/human_check/ with 20 pairs (generated vs real
    reference) and an empty human_check.csv with columns
    pair_id, same_writer_guess, confidence. Mark DEFERRED for me to rate.

GATE 5: automatic pass if all figures, tables and files above exist.

---------------------------------------------------------------------------
## STAGE 6 — Product and cleanup

6.1 Streamlit app: upload sample -> choose backend (baseline / VATr) ->
    type text -> render output, showing legibility and fidelity scores.
6.2 README: setup; data acquisition (Teklia/IAM-line, forms.txt source,
    Marti & Bunke 2002 citation); HWD and VATr citations; exact commands
    per stage; final results table.
6.3 Rewrite PROJECT_STATUS.md to reflect the final state.

GATE 6 (all must pass):
- app launches without errors (smoke-test headless)
- full test suite passes

---------------------------------------------------------------------------
## FINAL REPORT
When Stage 6 passes, write FINAL_REPORT.md: results table, figures, what
worked, what failed, every DEFERRED [HUMAN] item, and every place results
are labelled "subset" or "CPU-reduced".