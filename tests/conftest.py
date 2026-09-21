"""Unit tests build fake IAM corpora with their own writers, so they exercise the seeded split, not the canonical
VATr one that config.yaml selects (whose fixed writer lists cannot match a fake corpus)."""
import os

os.environ["IAM_SPLIT_SOURCE"] = "seeded"     # inherited by the subprocesses some tests launch
