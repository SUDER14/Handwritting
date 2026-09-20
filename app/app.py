"""Streamlit UI: Personalized Handwriting Generator (section 16 of the design brief).

Run with:
    streamlit run app/app.py

Workflow implemented:
    1. Upload a handwriting sample (PNG/JPG/JPEG) + type the word/text it contains.
    2. Preview the uploaded image + detected characters.
    3. Handwriting style analysis (slant, stroke width, height, width, spacing).
    4. Generate the full lowercase alphabet (baseline CV backend, or the trained
       neural style-encoder backend if a checkpoint exists).
    5. Preview the generated alphabet.
    6. Enter arbitrary text and generate a handwriting-style image of it.
    7. Download the result.
"""
from __future__ import annotations

import io
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.generator.alphabet_builder import AlphabetBuilder, AlphabetMetadata
from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural, recognize_with_cnn
from src.renderer.text_renderer import TextRenderer
from src.utils.config import load_config, resolve_path

st.set_page_config(page_title="Personalized Handwriting Generator", layout="wide")

CFG = load_config()
CHECKPOINT_PATH = resolve_path(CFG["training"]["checkpoint_path"])


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def glyph_to_display(img: np.ndarray) -> np.ndarray:
    """Invert ink=255-on-black to ink=dark-on-white for a natural on-screen look."""
    return 255 - img


def alphabet_strip(alphabet: dict[str, list[np.ndarray]], charset: str = string.ascii_lowercase) -> np.ndarray:
    """Compose all requested characters into one horizontal strip image for preview."""
    imgs = [glyph_to_display(alphabet[c][0]) for c in charset if c in alphabet and alphabet[c]]
    if not imgs:
        return np.full((60, 60), 255, dtype=np.uint8)
    h = max(im.shape[0] for im in imgs)
    pad_imgs = []
    for im in imgs:
        pad = h - im.shape[0]
        pad_imgs.append(cv2.copyMakeBorder(im, 0, pad, 4, 4, cv2.BORDER_CONSTANT, value=255))
    return np.hstack(pad_imgs)


def main():
    st.title("Personalized Handwriting Generator")
    st.caption(
        "Upload a small handwriting sample -> the system learns your style -> "
        "generates the letters you didn't write -> renders any text you type."
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"

    # -- 1. Upload -----------------------------------------------------------
    st.header("1. Upload handwriting sample")
    col_upload, col_label = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload Image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])
    with col_label:
        label_text = st.text_input("What word did you write? (ground truth, e.g. 'quick')", value="quick")
        cnn_checkpoint_exists = resolve_path(CFG["recognition"]["cnn_checkpoint_path"]).exists()
        auto_detect = st.checkbox(
            "Auto-detect the word with the trained CNN instead (experimental, open-vocabulary)",
            value=False,
            disabled=not cnn_checkpoint_exists,
            help=(
                "Train it first with `python scripts/train_cnn_recognizer.py` to enable this."
                if not cnn_checkpoint_exists else
                "Segmentation and per-character identity come from the trained CNN's own reading "
                "of the image (src/recognition/cnn_recognizer.py), not from the text typed above."
            ),
        )

    if uploaded is None:
        st.info("Upload a handwriting sample to begin. Tip: write a word with varied letters (e.g. 'quick').")
        return

    image = Image.open(io.BytesIO(uploaded.read()))
    image_bgr = pil_to_bgr(image)

    st.header("2. Preview")
    st.image(image, caption="Uploaded sample", width=400)

    label_text_clean = "".join(ch for ch in label_text.strip().lower() if ch.isalpha())
    if not label_text_clean:
        st.warning("Enter the word you wrote (letters only) to continue.")
        return

    with st.spinner("Analyzing handwriting..."):
        try:
            recognition_backend = "cnn" if auto_detect else None
            analysis = analyze_sample(image_bgr, label_text_clean, CFG, recognition_backend=recognition_backend)
        except Exception as e:  # noqa: BLE001 — surface any pipeline failure to the user
            st.error(f"Could not analyze the sample: {e}")
            return

    st.header("3. Detected characters")
    if auto_detect:
        st.caption("Detected automatically by the trained CNN recognizer — the typed word above was not used.")
    st.write(" ".join(analysis.chars))
    cols = st.columns(min(8, max(1, len(analysis.segmentation.glyphs))))
    for i, g in enumerate(analysis.segmentation.glyphs):
        with cols[i % len(cols)]:
            st.image(glyph_to_display(g.image), caption=analysis.chars[i], width=60)

    if not auto_detect:
        check = recognize_with_cnn(analysis, CFG)
        if check is None:
            st.info(
                "Recognition cross-check unavailable: no trained CNN checkpoint found. "
                "Run `python scripts/train_cnn_recognizer.py` to enable it."
            )
        elif check.typed_chars:
            st.subheader("Recognition check")
            st.caption(
                "Independent cross-check: does the trained CNN, looking only at pixels, read each "
                "segmented glyph the same way you typed it?"
            )
            mismatches = sum(1 for ok in check.agreement if not ok)
            if mismatches:
                st.warning(
                    f"The CNN disagrees with {mismatches}/{len(check.typed_chars)} character(s). This "
                    "usually means segmentation split or merged a glyph wrong — common where cursive "
                    "strokes touch — rather than the classifier being confused; check the crops above."
                )
            else:
                st.success("The CNN agrees with every character you typed.")
            check_cols = st.columns(min(8, max(1, len(check.typed_chars))))
            for i, (typed, pred, conf, ok) in enumerate(
                zip(check.typed_chars, check.predicted_chars, check.confidences, check.agreement)
            ):
                with check_cols[i % len(check_cols)]:
                    st.image(glyph_to_display(analysis.segmentation.glyphs[i].image), width=50)
                    marker = "match" if ok else "MISMATCH"
                    st.caption(f"typed '{typed}' / CNN '{pred}' ({conf * 100:.0f}%) — {marker}")

    st.header("4. Handwriting style analysis")
    sp = analysis.style_profile
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slant", f"{sp.mean_slant_deg:.1f} deg")
    m2.metric("Stroke width", f"{sp.mean_stroke_width_px:.1f} px")
    m3.metric("Avg. char height", f"{sp.mean_height_px:.0f} px")
    m4.metric("Avg. char width", f"{sp.mean_width_px:.0f} px")
    m5, m6, m7 = st.columns(3)
    m5.metric("Avg. spacing", f"{sp.mean_char_spacing_px:.1f} px")
    m6.metric("Curvature", f"{sp.mean_curvature_score:.2f}")
    m7.metric("Style feature dims", f"{len(sp.feature_vector)}")

    # -- 5. Generate alphabet -------------------------------------------------
    st.header("5. Generate alphabet")
    has_neural = CHECKPOINT_PATH.exists()
    backend = st.radio(
        "Generation backend",
        options=(["neural (trained conditional VAE)", "baseline (style-transformed reference font)"]
                 if has_neural else ["baseline (style-transformed reference font)"]),
        help="The neural backend needs a trained checkpoint (scripts/train_style_encoder.py).",
    )

    if st.button("Generate", type="primary"):
        with st.spinner("Generating personalized alphabet..."):
            if backend.startswith("neural"):
                alphabet = generate_alphabet_neural(analysis, checkpoint_path=str(CHECKPOINT_PATH))
            else:
                alphabet = generate_alphabet_baseline(analysis, CFG)
            st.session_state.alphabet = alphabet
            st.session_state.backend_used = backend

            builder = AlphabetBuilder(resolve_path("outputs/alphabets"))
            meta = AlphabetMetadata(
                generator_backend=backend,
                observed_chars=list(analysis.observed_glyphs.keys()),
                generated_chars=[c for c in string.ascii_lowercase if c not in analysis.observed_glyphs],
            )
            builder.save(alphabet, meta, st.session_state.session_id)

    if "alphabet" in st.session_state:
        st.header("6. Preview generated alphabet")
        st.image(alphabet_strip(st.session_state.alphabet), caption="a - z", use_container_width=True)
        st.caption(f"Backend used: {st.session_state.backend_used}")

        st.header("7. Enter text")
        text_input = st.text_input(
            "Text to render in your handwriting style", value="Artificial Intelligence is changing the world."
        )

        st.header("8. Generate handwriting")
        if st.button("Generate handwriting image"):
            renderer = TextRenderer(st.session_state.alphabet, CFG)
            rendered_bgr = renderer.render_to_bgr_on_white(text_input.lower())
            rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
            st.header("9. Output image")
            st.image(rendered_rgb, caption=text_input, use_container_width=True)

            out_path = resolve_path("outputs/generated_text") / f"{st.session_state.session_id}_{int(time.time())}.png"
            cv2.imwrite(str(out_path), rendered_bgr)
            buf = io.BytesIO()
            Image.fromarray(rendered_rgb).save(buf, format="PNG")
            st.download_button("Download PNG", data=buf.getvalue(), file_name="handwriting.png", mime="image/png")


if __name__ == "__main__":
    main()
