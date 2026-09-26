Superseded Stage-2 protocol run 20260922T190706Z (code 59bbb720). NOT quoted anywhere.
Reason: run_writer swallowed generation exceptions without logging; a burst of sub-second all-failed writers
(baseline K=5: 3 writers, glyph_vae K=1: 19 of 20 writers skipped at 02:39:57-59, see logs/protocol_stage2.log)
could not be diagnosed; the same writers generate fine when rerun alone (likely transient host OOM, ~0.9 GB free).
Also the coverage counters were cumulative across K. Fixed in the Task 2.2 commit and rerun.

Superseded Stage-2 rerun r2 20260926T184457Z (code 1c148084; "-dirty" = only ROADMAP_STATE.md/untracked scratch files,
src/ clean). NOT quoted anywhere. baseline K=1 scored 5 of 20 writers: 15 skipped at 00:26:51-56 local by host
out-of-memory (torch "DefaultCPUAllocator: not enough memory", OpenCV "(-4:Insufficient memory)"); the process was then
killed externally mid baseline K=5 (log ends 00:38 with no traceback; glyph_vae never ran). Root cause, measured:
the protocol process is flat at ~2.2-2.4 GB private (no per-call growth, K=5 profile), system commit was ~16 GB of a
24.7 GB limit without it; the OOM coincided with a concurrent Stage-3 prep download of VATr's 590 MB IAM-32.pickle
(finished 00:25:44, a duplicate .part copy left behind), i.e. a second memory-heavy job on this 7.8 GB host.
Also: the MemoryError retry never fired because torch/OpenCV OOMs are not Python MemoryError. Fixed in the next
commit (OOM classification + per-target and per-writer retry, coverage restored on retry, reference analysis cached
per writer -- bit-identical metrics, ~3x faster) and rerun with nothing else heavy running. Log: protocol_stage2_r2.log.
