# Human check (Task 5.4) -- STATUS: DEFERRED, waiting for your ratings

For each `pairs/pair_NN_A.png` (a line of handwriting, real or generated -- you are not told which) and `pairs/pair_NN_B_ref.png` (a REAL reference line), fill `human_check.csv`:

* `same_writer_guess`: `yes` if you think A is in the handwriting of whoever wrote B, else `no`
* `confidence`: 1 (guess) to 5 (certain)

20 pairs, 3 writers (172, 173, 179), K=1 samples from Task 3.6. **Do not open `answer_key.csv` before rating** (it lists the source of A and the truth). Mix: 8 baseline, 8 glyph_vae, 4 real held-out control lines; half the pairs are same-writer, half different-writer. VATr pairs will be added after the Colab run.
