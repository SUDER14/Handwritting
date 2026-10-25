"""Streamlit app (ROADMAP Task 6.1): give a handwriting line, pick a backend, type text, see the output with its scores.

    streamlit run app/app.py

Backends: real / baseline / glyph_vae. VATr appears as a switch that stays disabled (with the reason) unless the released
checkpoint AND a CUDA GPU are present. Next to every output: legibility (the frozen CRNN's CER on the output) and
fidelity (pinned writer-ID cosine distance and official HWD against the writer's real lines) -- the evaluation
protocol's instruments, not new ones. Lower is better for all three.

The first score loads the CRNN, the writer-ID net and the 500 MB HWD backbone (tens of seconds on CPU).
The pre-v6 alphabet-building UI is kept in app/legacy_app.py.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src import app_logic as L
from src.evaluation.harness import render_text_clean
from src.utils.config import load_config

st.set_page_config(page_title="Handwriting synthesis", layout="wide")
CFG = load_config()


@st.cache_resource(show_spinner="Loading the frozen scoring instruments (CRNN, writer-ID, HWD)...")
def instruments() -> L.Instruments:
    return L.Instruments(CFG)


@st.cache_data(show_spinner=False)
def demo_writer_ids() -> list[str]:
    return L.demo_writers(CFG)


@st.cache_data(show_spinner=False)
def demo_case(writer: str) -> L.DemoCase:
    return L.demo_case(writer, CFG)


def to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    st.title("Handwriting style synthesis")
    st.caption("Reference line -> backend -> text in that handwriting, scored with the evaluation protocol's frozen instruments "
               "(CRNN legibility, writer-ID and HWD fidelity). CPU-reduced research code, not a product.")

    st.header("1. Reference handwriting")
    source = st.radio("Source", ["IAM test-split demo writer", "Upload my own line"], horizontal=True, key="source")
    ref_gray, ref_text, real_lines, case = None, "", [], None
    if source.startswith("IAM"):
        try:
            writers = demo_writer_ids()
        except Exception as e:                                               # noqa: BLE001 -- data/iam not on this host
            st.error(f"IAM demo data is not available on this machine ({e}). Use 'Upload my own line'.")
            writers = []
        if writers:
            w = st.selectbox("Writer (the protocol's seeded 20-writer test subset)", writers, key="writer")
            case = demo_case(w)
            ref_gray, ref_text, real_lines = case.ref_gray, case.ref_text, case.heldout
            st.image(ref_gray, caption=f"Reference line (K=1) of writer {w}: \"{ref_text}\"", use_container_width=True)
            st.caption(f"Scores below compare outputs with this writer's {len(real_lines)} REAL held-out lines "
                       "(the protocol's comparison set).")
    else:
        up = st.file_uploader("Handwriting line (PNG / JPG)", type=["png", "jpg", "jpeg"], key="upload")
        ref_text = st.text_input("What does the line say? (needed to find letters)", key="ref_text")
        if up is not None:
            ref_gray = to_gray(Image.open(io.BytesIO(up.read())))
            real_lines = [ref_gray]
            st.image(ref_gray, caption="Your line", use_container_width=True)
            st.caption("Scores below compare outputs with your uploaded line (only one real line is available).")

    st.header("2. Backend")
    backend = st.radio("Backend", L.BACKENDS, horizontal=True, key="backend",
                       help="real = one of the writer's own held-out lines (the upper bound); baseline = font glyphs warped to "
                            "the writer's style; glyph_vae = EMNIST VAE ablation.")
    vatr_ok, vatr_msg = L.vatr_status(CFG)
    use_vatr = st.toggle("Use VATr instead", value=False, disabled=not vatr_ok, key="use_vatr")
    (st.caption if vatr_ok else st.warning)(vatr_msg)
    if use_vatr:
        backend = "vatr"
    if backend == "real" and case is None:
        st.info("'real' needs a held-out line of the same writer: it is only available for the IAM demo writers.")

    st.header("3. Text")
    default = case.heldout_texts[0] if case is not None and backend == "real" else "the quick brown fox"
    text = st.text_input("Text to write (letters and spaces; anything else is dropped)", value=default, key="text")
    clean = render_text_clean(text)
    if clean != text.strip().lower():
        st.caption(f"Will be rendered as: \"{clean}\"")

    if st.button("Generate", type="primary", key="generate"):
        if ref_gray is None or not (ref_text.strip() or backend == "real"):
            st.warning("Provide a reference line (and, for an upload, its text) first.")
            return
        if not clean and backend != "real":
            st.warning("The text has no letters.")
            return
        try:
            with st.spinner(f"Generating with {backend}..."):
                if backend == "real":
                    if case is None:
                        st.error("'real' is only available for the IAM demo writers.")
                        return
                    out, shown_text = case.heldout[0], case.heldout_texts[0]
                else:
                    out, shown_text = L.generate_line(backend, ref_gray, ref_text, clean, CFG), clean
            if out is None:
                st.error("The backend produced no output for this text.")
                return
            st.header("4. Output")
            st.image(out, caption=f"{backend}: \"{shown_text}\"", use_container_width=True)
            st.download_button("Download PNG", data=png_bytes(out), file_name=f"{backend}.png", mime="image/png")
            with st.spinner("Scoring with the frozen instruments..."):
                eval_lines = real_lines[1:] if backend == "real" and len(real_lines) > 1 else real_lines
                sc = instruments().score(out, shown_text, eval_lines)
            c1, c2, c3 = st.columns(3)
            c1.metric("Legibility: CER (lower = better)", f"{sc['cer']:.3f}")
            c2.metric("Fidelity: writer-ID distance", f"{sc['wid_dist']:.3f}")
            c3.metric("Fidelity: HWD", f"{sc['hwd']:.2f}")
            st.caption(f"CRNN read: \"{sc['hyp']}\" | compared with {sc['n_real']} real line(s). For orientation, the protocol's "
                       "20-writer means (results/protocol_table.csv) are in the README.")
        except Exception as e:                                                   # noqa: BLE001 -- show, never crash the page
            st.error(f"Generation or scoring failed: {type(e).__name__}: {e}")


main()
