Superseded Stage-2 protocol run 20260922T190706Z (code 59bbb720). NOT quoted anywhere.
Reason: run_writer swallowed generation exceptions without logging; a burst of sub-second all-failed writers
(baseline K=5: 3 writers, glyph_vae K=1: 19 of 20 writers skipped at 02:39:57-59, see logs/protocol_stage2.log)
could not be diagnosed; the same writers generate fine when rerun alone (likely transient host OOM, ~0.9 GB free).
Also the coverage counters were cumulative across K. Fixed in the Task 2.2 commit and rerun.
