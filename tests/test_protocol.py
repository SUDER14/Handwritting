"""Protocol plumbing (src/evaluation/protocol.py) with fake instruments: no models, no IAM, no network."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation import protocol as P

S = lambda i, w, text: SimpleNamespace(sample_id=f"{w}-{i:02d}", writer_id=w, transcription=text)   # noqa: E731
TEXTS = ["the quick brown fox jumps", "over the lazy dog again", "pack my box with five dozen",
         "liquor jugs sphinx of black quartz", "how vexingly quick daft zebras", "jump", "waltz bad nymph for quick"]


def _samples(w="007"):
    return [S(i, w, t) for i, t in enumerate(TEXTS)]


def test_refs_and_targets_are_disjoint_seeded_and_k_dependent():
    a = P.split_lines(_samples(), 2, 3, 12, 1337, "007")
    b = P.split_lines(_samples(), 2, 3, 12, 1337, "007")
    assert [s.sample_id for s in a[0]] == [s.sample_id for s in b[0]]                     # deterministic
    refs, targets = a
    assert not ({r.sample_id for r in refs} & {t.sample_id for t in targets}) and len(refs) == 2 and len(targets) == 3
    assert all(len(r.transcription) >= 12 for r in refs)                                   # min_reference_letters
    assert P.split_lines(_samples()[:2], 2, 3, 12, 1337, "007") is None                    # not enough lines for K + 1


def test_oov_means_a_target_letter_absent_from_all_reference_texts():
    assert not P.is_oov("the fox", ["the quick brown fox"])
    assert P.is_oov("zebra", ["the quick brown fox"])            # z, e.. z absent
    assert not P.is_oov("zebra", ["zebra x", "abc"])
    assert P.is_oov("axe", ["a", "x"]) and not P.is_oov("axe", ["a", "x", "e"])


def test_choose_writers_is_a_deterministic_subset_of_writers_with_enough_lines():
    ds = SimpleNamespace(samples=[S(i, f"{w:03d}", "x") for w in range(10) for i in range(3 + w % 3)])
    a = P.choose_writers(ds, 4, 1, 4)
    assert a == P.choose_writers(ds, 4, 1, 4) and len(a) == 4
    assert all(sum(s.writer_id == w for s in ds.samples) >= 4 for w in a)


def test_aggregate_is_mean_over_writers_and_oov_counts_only_writers_that_have_one():
    def row(cer, oov):
        m = lambda c: {"cer": c, "wid_dist": c / 2, "hwd": c * 10, "n_lines": 4}       # noqa: E731
        return {"all": m(cer), "oov": m(oov) if oov is not None else None, "n_failed": 1}
    agg = P.aggregate([row(0.2, 0.4), row(0.4, None)])
    assert agg["all_cer"] == pytest.approx(0.3) and agg["all_hwd"] == pytest.approx(3.0)
    assert agg["oov_cer"] == pytest.approx(0.4) and agg["oov_n_writers"] == 1 and agg["n_failed_generations"] == 2


def test_run_writer_real_row_and_backend_row_with_fake_instruments():
    class DS:
        samples = _samples()
        def read_native(self, i):
            img = np.full((64, 200), 255, np.uint8); img[20:40, 20:180] = 0
            return img
    class Scorer:
        def read(self, grays): return ["the quick"] * len(grays)
        def embed(self, g): return np.array([1.0, 0.0, 0.1])
        def hwd_entry(self, g, key=None): return (np.ones(4) * 3, 3)
    class Backend:
        name = "fake"
        def generate(self, refs, text):
            assert len(refs) == 2 and refs[0].gray.ndim == 2
            if text.startswith("liquor"):
                return np.full((64, 100), 255, np.uint8)             # blank generation = a counted failure
            img = np.full((64, 100), 255, np.uint8); img[20:40, 10:90] = 0
            return img
    cfg = {"targets_per_writer": 4, "min_reference_letters": 12, "max_target_chars": 40}
    ds = DS()
    real = P.run_writer(Scorer(), None, ds, ds.samples, 2, cfg, 1, "007")
    assert real["all"]["hwd"] == 0.0 and real["all"]["wid_dist"] == pytest.approx(0.0, abs=1e-9)
    fake = P.run_writer(Scorer(), Backend(), ds, ds.samples, 2, cfg, 1, "007")
    assert fake["n_failed"] + fake["all"]["n_lines"] == fake["n_targets"]
    assert fake["n_failed"] == sum(t.startswith("liquor") for t in [ds.samples[i].transcription for i in range(len(ds.samples))
                                                                    if ds.samples[i].sample_id in fake["target_ids"]])


def test_generation_exceptions_are_recorded_and_memoryerror_is_retried_once():
    class DS:
        samples = _samples()
        def read_native(self, i):
            img = np.full((64, 200), 255, np.uint8); img[20:40, 20:180] = 0
            return img
    class Scorer:
        def read(self, grays): return ["x"] * len(grays)
        def embed(self, g): return np.array([1.0, 0.0])
        def hwd_entry(self, g, key=None): return (np.ones(4), 1)
    class Flaky:
        name = "flaky"
        def __init__(self): self.calls = 0
        def generate(self, refs, text):
            self.calls += 1
            if self.calls == 1:
                raise MemoryError("transient")                        # retried -> succeeds
            if text.startswith("liquor"):
                raise ValueError("boom")                              # not retried -> counted failure
            img = np.full((64, 100), 255, np.uint8); img[20:40, 10:90] = 0
            return img
    cfg = {"targets_per_writer": 4, "min_reference_letters": 12, "max_target_chars": 40}
    ds, be = DS(), Flaky()
    r = P.run_writer(Scorer(), be, ds, ds.samples, 2, cfg, 1, "007")
    n_liquor = sum(ds.samples[i].transcription.startswith("liquor") for i in range(len(ds.samples))
                   if ds.samples[i].sample_id in r["target_ids"])
    assert r["n_failed"] == n_liquor
    assert r["gen_errors"][0].startswith("OutOfMemory[MemoryError]") and sum(e.startswith("ValueError") for e in r["gen_errors"]) == n_liquor


def test_library_out_of_memory_errors_are_recognised_as_oom():
    # the exact messages that skipped 15/20 writers in the 2026-09-27 Stage-2 rerun (torch CPU allocator, OpenCV)
    torch_oom = RuntimeError("[enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: "
                             "you tried to allocate 7045120 bytes.")
    cv_oom = Exception("OpenCV(5.0.0) alloc.cpp:73: error: (-4:Insufficient memory) Failed to allocate 6393600 bytes "
                       "in function 'cv::OutOfMemoryError'")
    assert P.is_out_of_memory(torch_oom) and P.is_out_of_memory(cv_oom) and P.is_out_of_memory(MemoryError())
    assert not P.is_out_of_memory(ValueError("boom"))


def test_oom_from_a_library_is_retried_once_not_counted_as_a_failed_generation():
    class DS:
        samples = _samples()
        def read_native(self, i):
            img = np.full((64, 200), 255, np.uint8); img[20:40, 20:180] = 0
            return img
    class Scorer:
        def read(self, grays): return ["x"] * len(grays)
        def embed(self, g): return np.array([1.0, 0.0])
        def hwd_entry(self, g, key=None): return (np.ones(4), 1)
    class CvFlaky:
        name = "cvflaky"
        def __init__(self): self.calls = 0
        def generate(self, refs, text):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("(-4:Insufficient memory) Failed to allocate 100 bytes")
            img = np.full((64, 100), 255, np.uint8); img[20:40, 10:90] = 0
            return img
    cfg = {"targets_per_writer": 4, "min_reference_letters": 12, "max_target_chars": 40}
    ds = DS()
    r = P.run_writer(Scorer(), CvFlaky(), ds, ds.samples, 2, cfg, 1, "007")
    assert r["n_failed"] == 0 and len(r["gen_errors"]) == 1 and r["gen_errors"][0].startswith("OutOfMemory[RuntimeError]")


def test_generation_whose_ink_vanishes_on_canonicalisation_is_a_failed_line_not_a_lost_writer():
    # 2026-09-27 glyph_vae K=5 writer 547: one sparse generation passed the raw ink check, its 64-px canonical crop was
    # blank, embed() raised and the whole writer was skipped. Real Scorer.crop + the real embed() blank-check are used.
    class DS:
        samples = _samples()
        def read_native(self, i):
            img = np.full((64, 200), 255, np.uint8); img[20:40, 20:180] = 0
            return img
    class Emb:
        def embed(self, line):
            if not np.any(np.asarray(line) > 127):
                raise ValueError("blank image; cannot embed")
            return np.array([1.0, 0.0])
    class Rec:
        def transcribe(self, crops): return ["x"] * len(crops)
    class Feats:
        def line(self, g): return (np.ones(4), 1)
    class Sparse:
        name = "sparse"
        def generate(self, refs, text):
            img = np.full((260, 1600), 255, np.uint8)
            if text.startswith("liquor"):
                img[100, 800] = 0                                    # one dark pixel: raw check passes, crop is blank
            else:
                img[80:180, 100:1500] = 0
            return img
    scorer = P.Scorer(Rec(), Emb(), Feats(), 64)
    faint = np.full((260, 1600), 255, np.uint8); faint[100, 800] = 0
    assert np.any(faint < 128) and not scorer.scorable(faint)
    cfg = {"targets_per_writer": 4, "min_reference_letters": 12, "max_target_chars": 40}
    ds = DS()
    r = P.run_writer(scorer, Sparse(), ds, ds.samples, 2, cfg, 1, "007")
    n_liquor = sum(ds.samples[i].transcription.startswith("liquor") for i in range(len(ds.samples))
                   if ds.samples[i].sample_id in r["target_ids"])
    assert n_liquor == 1 and r is not None and r["n_failed"] == 1 and r["all"]["n_lines"] == r["n_targets"] - 1
