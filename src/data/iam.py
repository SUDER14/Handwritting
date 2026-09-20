"""IAM Handwriting Database loader with a writer-disjoint, seed-deterministic split.

Expected on-disk layout (the standard IAM distribution; the dataset is licensed
and is not part of this repo -- put it under `data.iam.root` in config.yaml)::

    <root>/ascii/forms.txt      form id -> writer id
    <root>/ascii/lines.txt      line id, status, ..., transcription ('|' = word gap)
    <root>/ascii/words.txt      word id, status, ..., tag, transcription
    <root>/lines/<a01>/<a01-000u>/<a01-000u-00>.png
    <root>/words/<a01>/<a01-000u>/<a01-000u-00-00>.png

Why a custom split instead of IAM's official ones: the goal here is
*writer*-disjoint evaluation of few-shot style transfer, and it must be
reproducible and auditable. So:

* The split is over WRITERS, taken from `forms.txt` (every writer the metadata
  knows about, whether or not their images are on disk). That keeps the split
  stable if only part of the corpus is downloaded.
* It is a pure function of (`split_seed`, the set of writer IDs, the
  fractions): writers are ranked by ``sha256(f"{seed}:{writer_id}")`` -- no
  dependence on Python's RNG implementation or on file ordering -- then the
  first ``n_test`` go to test, the next ``n_val`` to val, the rest to train.
* The writer lists are written to ``<splits_dir>/{train,val,test}.json`` and
  re-verified on every load, so a drifting config or hand-edited file fails loudly.

Images are returned in IAM's native polarity (dark ink on light paper,
uint8 grayscale), i.e. the same kind of input `Preprocessor.process` expects.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
SPLIT_ALGORITHM = (
    "rank writers by sha256('<seed>:<writer_id>'); first n_test -> test, "
    "next n_val -> val, remainder -> train"
)
UNITS = ("lines", "words")


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def _data_lines(path: Path):
    """Yield non-comment, non-blank lines of an IAM ascii file."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            yield line


def parse_forms(forms_txt: Path) -> dict[str, str]:
    """`forms.txt` -> {form_id: writer_id}. Only the first two fields are used."""
    forms: dict[str, str] = {}
    for line in _data_lines(forms_txt):
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed forms.txt line: {line!r}")
        forms[parts[0]] = parts[1]
    return forms


def parse_lines(lines_txt: Path) -> list[dict]:
    """`lines.txt` -> [{id, status, transcription}]. Format:
    ``a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop``.
    """
    rows = []
    for line in _data_lines(lines_txt):
        parts = line.split(None, 8)
        if len(parts) < 9:
            raise ValueError(f"Malformed lines.txt line: {line!r}")
        rows.append({"id": parts[0], "status": parts[1], "transcription": parts[8].replace("|", " ")})
    return rows


def parse_words(words_txt: Path) -> list[dict]:
    """`words.txt` -> [{id, status, transcription}]. Format:
    ``a01-000u-00-00 ok 154 408 768 27 51 AT A``.
    """
    rows = []
    for line in _data_lines(words_txt):
        parts = line.split(None, 8)
        if len(parts) < 9:
            raise ValueError(f"Malformed words.txt line: {line!r}")
        rows.append({"id": parts[0], "status": parts[1], "transcription": parts[8]})
    return rows


def form_id_of(sample_id: str, unit: str) -> str:
    """'a01-000u-00' (line) -> 'a01-000u';  'a01-000u-00-00' (word) -> 'a01-000u'."""
    return sample_id.rsplit("-", 1 if unit == "lines" else 2)[0]


def image_path_for(root: Path, unit: str, sample_id: str) -> Path:
    form_id = form_id_of(sample_id, unit)
    return root / unit / form_id.split("-")[0] / form_id / f"{sample_id}.png"


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------

def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize to a fixed height, width scaled to preserve aspect ratio (>= 1 px)."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Cannot resize an empty image of shape {image.shape}")
    new_w = max(1, int(round(w * height / h)))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, height), interpolation=interp)


# ---------------------------------------------------------------------------
# Writer-disjoint split
# ---------------------------------------------------------------------------

def _validate_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"split_fractions must have exactly keys {SPLIT_NAMES}, got {sorted(fractions)}")
    if any(v <= 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must be positive and sum to 1, got {fractions}")


def make_writer_splits(writer_ids, seed: int, fractions: dict[str, float]) -> dict[str, list[str]]:
    """Deterministic writer-disjoint split. See the module docstring for the algorithm."""
    _validate_fractions(fractions)
    writers = sorted(set(writer_ids))
    n = len(writers)
    if n < 3:
        raise ValueError(f"Need at least 3 writers for a train/val/test split, got {n}")
    ranked = sorted(writers, key=lambda w: hashlib.sha256(f"{seed}:{w}".encode()).hexdigest())
    n_test = max(1, round(n * fractions["test"]))
    n_val = max(1, round(n * fractions["val"]))
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} writers is too few for fractions {fractions}")
    return {
        "test": sorted(ranked[:n_test]),
        "val": sorted(ranked[n_test:n_test + n_val]),
        "train": sorted(ranked[n_test + n_val:]),
    }


def _split_payload(name: str, writer_ids: list[str], seed: int, fractions: dict[str, float], source: str) -> dict:
    return {
        "split": name,
        "seed": seed,
        "fractions": fractions,
        "algorithm": SPLIT_ALGORITHM,
        "source": source,
        "num_writers": len(writer_ids),
        "writer_ids": writer_ids,
    }


def write_split_files(splits: dict[str, list[str]], splits_dir: Path, seed: int,
                      fractions: dict[str, float], source: str = "ascii/forms.txt") -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        payload = _split_payload(name, splits[name], seed, fractions, source)
        (splits_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_split_files(splits_dir: Path) -> tuple[dict[str, list[str]], dict]:
    """Returns ({split: writer_ids}, {seed, fractions}) as stored on disk."""
    splits, meta = {}, {}
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits[name] = list(payload["writer_ids"])
        meta[name] = {"seed": payload["seed"], "fractions": payload["fractions"]}
    seeds = {m["seed"] for m in meta.values()}
    fracs = {json.dumps(m["fractions"], sort_keys=True) for m in meta.values()}
    if len(seeds) != 1 or len(fracs) != 1:
        raise ValueError(f"Split files in {splits_dir} disagree with each other on seed/fractions")
    return splits, next(iter(meta.values()))


def get_writer_splits(writer_ids, seed: int, fractions: dict[str, float], splits_dir: Path,
                      regenerate: bool = False) -> dict[str, list[str]]:
    """Return the writer split, creating the JSON files on first use.

    If the files already exist they are *verified*, not trusted: they must equal
    what (seed, fractions, writer_ids) produce now. A mismatch raises instead of
    silently evaluating on a different split than the one that is on record.
    """
    expected = make_writer_splits(writer_ids, seed, fractions)
    files_exist = all((splits_dir / f"{n}.json").exists() for n in SPLIT_NAMES)
    if regenerate or not files_exist:
        write_split_files(expected, splits_dir, seed, fractions)
        logger.info("Wrote writer-disjoint split files to %s (seed=%s)", splits_dir, seed)
        return expected
    on_disk, meta = read_split_files(splits_dir)
    if meta["seed"] != seed or meta["fractions"] != fractions:
        raise ValueError(
            f"{splits_dir} was made with seed={meta['seed']} fractions={meta['fractions']} but config has "
            f"seed={seed} fractions={fractions}. Re-run with regenerate=True (scripts/check_split.py --regenerate) "
            "if the change is intentional."
        )
    if on_disk != expected:
        raise ValueError(
            f"{splits_dir}/*.json do not match the split derived from seed={seed} and the writers in forms.txt "
            "(edited by hand, or forms.txt changed). Refusing to continue; regenerate deliberately if intended."
        )
    return on_disk


# ---------------------------------------------------------------------------
# Corpus + dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IAMSample:
    sample_id: str
    form_id: str
    writer_id: str
    image_path: Path
    transcription: str
    unit: str  # "lines" | "words"


class IAMCorpus:
    """Index of IAM samples grouped by writer (via forms.txt)."""

    def __init__(self, root: str | Path | None = None, unit: str = "lines", include_err: bool = False):
        if unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
        cfg = load_config()["data"]["iam"]
        self.root = Path(root) if root is not None else resolve_path(cfg["root"])
        self.unit = unit
        ascii_dir = self.root / "ascii"
        forms_txt, meta_txt = ascii_dir / "forms.txt", ascii_dir / f"{unit}.txt"
        for p in (forms_txt, meta_txt):
            if not p.exists():
                raise FileNotFoundError(
                    f"IAM metadata not found: {p}. Place the IAM database under {self.root} "
                    "(see the layout in src/data/iam.py)."
                )

        self.form_to_writer = parse_forms(forms_txt)
        self.all_writer_ids = sorted(set(self.form_to_writer.values()))
        rows = parse_lines(meta_txt) if unit == "lines" else parse_words(meta_txt)

        self.skipped = {"status_err": 0, "unknown_form": 0, "missing_image": 0}
        self.samples_by_writer: dict[str, list[IAMSample]] = defaultdict(list)
        for row in rows:
            if row["status"] != "ok" and not include_err:
                self.skipped["status_err"] += 1
                continue
            form_id = form_id_of(row["id"], unit)
            writer = self.form_to_writer.get(form_id)
            if writer is None:
                self.skipped["unknown_form"] += 1
                continue
            path = image_path_for(self.root, unit, row["id"])
            if not path.exists():
                self.skipped["missing_image"] += 1
                continue
            self.samples_by_writer[writer].append(
                IAMSample(row["id"], form_id, writer, path, row["transcription"], unit)
            )
        for samples in self.samples_by_writer.values():
            samples.sort(key=lambda s: s.sample_id)
        if any(self.skipped.values()):
            logger.info("IAM %s: skipped %s", unit, self.skipped)

    @property
    def num_samples(self) -> int:
        return sum(len(v) for v in self.samples_by_writer.values())


class IAMDataset:
    """One split of IAM. Indexing yields ``(image, transcription, writer_id)``.

    ``image`` is uint8 grayscale, resized to ``image_height`` rows with the
    width scaled to preserve aspect ratio (so widths vary). Use
    ``read_native(i)`` for the untouched image.
    """

    def __init__(self, samples: list[IAMSample], writer_ids: list[str], name: str, image_height: int):
        self.samples = samples
        self.writer_ids = writer_ids
        self.name = name
        self.image_height = image_height

    def __len__(self) -> int:
        return len(self.samples)

    def read_native(self, index: int) -> np.ndarray:
        path = self.samples[index].image_path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read IAM image {path}")
        return image

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, str]:
        sample = self.samples[index]
        return resize_to_height(self.read_native(index), self.image_height), sample.transcription, sample.writer_id

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_iam_splits(unit: str = "lines", config: dict | None = None, root: str | Path | None = None,
                    splits_dir: str | Path | None = None, regenerate: bool = False,
                    include_err: bool = False) -> dict[str, IAMDataset]:
    """Build {train, val, test} IAMDatasets with the auditable writer-disjoint split."""
    cfg = (config or load_config())["data"]["iam"]
    corpus = IAMCorpus(root=root, unit=unit, include_err=include_err)
    splits_path = Path(splits_dir) if splits_dir is not None else resolve_path(cfg["splits_dir"])
    writer_split = get_writer_splits(
        corpus.all_writer_ids, cfg["split_seed"], cfg["split_fractions"], splits_path, regenerate=regenerate
    )
    datasets = {}
    for name in SPLIT_NAMES:
        samples = [s for w in writer_split[name] for s in corpus.samples_by_writer.get(w, [])]
        datasets[name] = IAMDataset(samples, writer_split[name], name, cfg["image_height"])
    return datasets


def load_iam(split: str, unit: str = "lines", **kwargs) -> IAMDataset:
    """Convenience: one split by name ('train' | 'val' | 'test')."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    return load_iam_splits(unit=unit, **kwargs)[split]
