import numpy as np
import pytest
import torch

from src.recognition.crnn import CRNN, Recognizer, build_charset, cer_wer, encode, greedy_decode, pad_batch


def test_cer_wer_exact_arithmetic():
    cer, wer = cer_wer(["the cat", "sat"], ["the cut", "sit down"])
    assert cer == pytest.approx(7 / 10)              # 1 (cat->cut) + 1 sub + 5 inserted " down" = 7 edits over 7 + 3 ref chars
    assert wer == pytest.approx(3 / 3)               # "cat"->"cut" wrong, "sat"->"sit" wrong, "down" inserted


def test_greedy_ctc_decode_collapses_repeats_and_drops_blanks():
    charset = ["a", "b"]                             # ids: blank 0, a 1, b 2
    path = [1, 1, 0, 1, 2, 2, 0, 0]                  # a a _ a b b _ _  ->  "aab"
    logits = torch.full((len(path), 1, 3), -10.0)
    for t, k in enumerate(path):
        logits[t, 0, k] = 10.0
    assert greedy_decode(logits, torch.tensor([len(path)]), charset) == ["aab"]
    assert greedy_decode(logits, torch.tensor([3]), charset) == ["a"]         # only the first 3 steps count


def test_unseen_characters_are_dropped_from_targets_and_shapes_are_w_over_4():
    cs = build_charset(["ab", "ba"])
    assert encode("abz", {c: i + 1 for i, c in enumerate(cs)}) == [1, 2]
    x, lengths = pad_batch([np.zeros((64, 300), np.uint8), np.zeros((64, 200), np.uint8)])
    out = CRNN(len(cs))(x)
    assert out.shape == (75, 2, len(cs) + 1) and lengths.tolist() == [75, 50]
    assert Recognizer(CRNN(len(cs)), cs).transcribe([np.zeros((64, 120), np.uint8)]) is not None
