"""Builds a tiny fake IAM directory tree (same layout and file formats as the real
database) so the loader, split and evaluation code can be exercised without the
licensed dataset. Images are rendered with OpenCV's built-in Hershey font, so no
system fonts are needed. NOT handwriting -- structure only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SENTENCES = [
    "a move to stop mr gaitskell from",
    "nominating any more labour life peers",
    "is to be made at a meeting of labour",
    "mps tomorrow the quick brown fox",
    "jumps over the lazy dog again and again",
    "she sells sea shells by the sea shore",
]


def _render(text: str, scale: float, thickness: int) -> np.ndarray:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    img = np.full((h + base + 24, w + 24), 255, dtype=np.uint8)
    cv2.putText(img, text, (12, h + 12), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness, cv2.LINE_AA)
    return img


def build_fake_iam(root: Path, n_writers: int = 12, forms_per_writer: int = 2, lines_per_form: int = 3) -> dict:
    """Create <root>/{ascii,lines,words}. Returns bookkeeping for assertions:
    {"writers": [...], "form_to_writer": {...}, "line_ids_by_writer": {...}, "n_err": int, "n_missing": int}
    """
    root = Path(root)
    (root / "ascii").mkdir(parents=True, exist_ok=True)
    writers = [f"{i * 7 + 3:03d}" for i in range(n_writers)]
    form_to_writer: dict[str, str] = {}
    line_rows, word_rows = [], []
    line_ids_by_writer: dict[str, list[str]] = {w: [] for w in writers}
    n_err = n_missing = 0

    form_counter = 0
    for wi, writer in enumerate(writers):
        scale = 0.6 + 0.08 * (wi % 5)
        thick = 1 + wi % 2
        for _ in range(forms_per_writer):
            form_id = f"a{form_counter // 10 + 1:02d}-{form_counter:03d}u"
            form_counter += 1
            form_to_writer[form_id] = writer
            prefix = form_id.split("-")[0]
            for li in range(lines_per_form):
                line_id = f"{form_id}-{li:02d}"
                text = SENTENCES[(wi + li + form_counter) % len(SENTENCES)]
                status = "ok"
                # One deliberately bad ("err") line and one whose image file is absent.
                if wi == 0 and form_counter == 1 and li == 0:
                    status = "err"
                    n_err += 1
                line_rows.append(f"{line_id} {status} 154 19 408 746 1661 89 {text.replace(' ', '|')}")
                line_dir = root / "lines" / prefix / form_id
                line_dir.mkdir(parents=True, exist_ok=True)
                if wi == 1 and form_counter == 3 and li == 1:
                    n_missing += 1  # metadata row without an image file
                else:
                    cv2.imwrite(str(line_dir / f"{line_id}.png"), _render(text, scale, thick))
                    if status == "ok":
                        line_ids_by_writer[writer].append(line_id)
                word_dir = root / "words" / prefix / form_id
                word_dir.mkdir(parents=True, exist_ok=True)
                for k, word in enumerate(text.split()[:3]):
                    word_id = f"{line_id}-{k:02d}"
                    word_rows.append(f"{word_id} ok 154 408 768 27 51 NN {word}")
                    cv2.imwrite(str(word_dir / f"{word_id}.png"), _render(word, scale, thick))

    # A line whose form is not listed in forms.txt must be ignored, not crash.
    line_rows.append("z99-999z-00 ok 154 19 408 746 1661 89 orphan|line")

    header = "#--- fake IAM metadata for tests ---#\n#\n"
    (root / "ascii" / "forms.txt").write_text(
        header + "".join(f"{f} {w} 2 8 45\n" for f, w in form_to_writer.items()), encoding="utf-8")
    (root / "ascii" / "lines.txt").write_text(header + "\n".join(line_rows) + "\n", encoding="utf-8")
    (root / "ascii" / "words.txt").write_text(header + "\n".join(word_rows) + "\n", encoding="utf-8")
    return {"writers": writers, "form_to_writer": form_to_writer, "line_ids_by_writer": line_ids_by_writer,
            "n_err": n_err, "n_missing": n_missing}
