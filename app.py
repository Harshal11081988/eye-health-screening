"""
Eye Health Screening — Cataract Opacity & Pupil Symmetry Indicators
------------------------------------------------------------------------
Two photo-based screening indicators from a normal frontal face photo:

1. Cataract opacity indicator — flags visible pupil clouding, the way
   ADVANCED cataracts actually present in a normal photo (leukocoria).
2. Pupil symmetry (anisocoria) indicator — flags size differences
   between left/right pupils, a real neuro-ophthalmological sign.

⚠️ NOT a medical device or diagnostic tool. Cannot detect early-stage
cataracts (which need a slit-lamp exam to see), cannot detect
glaucoma (which requires optic nerve/pressure measurement no photo
can provide), and makes NO attempt at myopia/hyperopia/prescription
estimation (refractive error cannot be derived from any photo — see
README). See the disclaimers below before using this for anything
beyond curiosity.

Run locally with: streamlit run app.py
"""

import os
import tempfile
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from eye_analysis import analyze_eye_photo

st.set_page_config(page_title="Eye Health Screening", page_icon="👁️", layout="wide")


def draw_annotations(image_bgr, result):
    """Draw detected eye boxes and pupil circles on a copy of the image for display."""
    annotated = image_bgr.copy()
    detection = result["detection"]

    for bbox, color in [(detection["left_eye_bbox"], (255, 140, 0)), (detection["right_eye_bbox"], (0, 200, 255))]:
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    for pupil, bbox in [(result["left_pupil"], detection["left_eye_bbox"]), (result["right_pupil"], detection["right_eye_bbox"])]:
        if pupil is not None:
            ex, ey = bbox[0], bbox[1]
            cx, cy, r = pupil
            cv2.circle(annotated, (ex + cx, ey + cy), r, (0, 255, 0), 2)
            cv2.circle(annotated, (ex + cx, ey + cy), 2, (0, 0, 255), 3)

    return annotated


def main():
    st.title("👁️ Eye Health Screening Indicators")
    st.caption(
        "Two photo-based indicators computed from a normal front-facing photo: "
        "pupil clouding (cataract-related) and pupil size symmetry."
    )

    st.error(
        "⚠️ **This is not a medical device or diagnostic tool.** It cannot detect "
        "early-stage cataracts (which require a slit-lamp eye exam), cannot detect "
        "glaucoma (which requires optic nerve imaging and eye pressure measurement — "
        "no photo can provide either), and makes **no attempt at estimating a glasses "
        "prescription** (refractive error cannot be derived from any photo). If you "
        "have any eye health concerns, please see an eye doctor."
    )

    with st.expander("📋 What this can and can't do — please read"):
        st.markdown(
            "**Cataract opacity indicator:**\n"
            "- Can only flag *visible* pupil clouding (whitish/cloudy appearance), "
            "which is how *advanced* cataracts present in an ordinary photo\n"
            "- Cannot detect early-stage cataracts — those require a slit-lamp "
            "exam to see, which no phone camera can replicate\n"
            "- A low score does NOT mean your eyes are healthy — it only means "
            "no *advanced, visible* clouding was detected in this photo\n\n"
            "**Pupil symmetry indicator:**\n"
            "- Compares left/right pupil size in the same photo\n"
            "- Real anisocoria (pupil size difference) can be a meaningful "
            "neurological sign, but **~20% of people have mild, completely "
            "harmless physiological anisocoria** — a difference alone is not "
            "cause for alarm\n"
            "- Lighting affects pupil size (they constrict in bright light, "
            "dilate in dim light) — inconsistent lighting between eyes in the "
            "same photo can create a false asymmetry reading\n\n"
            "**What this deliberately does NOT do:**\n"
            "- No glaucoma detection (needs optic nerve imaging — impossible from a photo)\n"
            "- No myopia/hyperopia/prescription number (refractive error isn't "
            "visible in any photo — it requires measuring how light focuses "
            "inside a living, focusing eye)\n\n"
            "**Photo tips:** even, bright lighting on both eyes equally, "
            "looking straight at the camera, eyes fully open, no glasses glare."
        )

    uploaded_file = st.file_uploader("Upload a clear, front-facing photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a photo above to get started.")
        return

    image_pil = Image.open(uploaded_file).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    try:
        with st.spinner("Analyzing photo..."):
            result = analyze_eye_photo(image_bgr)
    except ValueError as e:
        st.error(str(e))
        st.image(image_pil, caption="Uploaded photo", use_container_width=True)
        return

    annotated = draw_annotations(image_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader("Detected eyes & pupils")
        st.image(annotated_rgb, use_container_width=True)
        st.caption("Orange/cyan boxes: detected eye regions. Green circles: detected pupils.")

    with col_results:
        st.subheader("Results")

        if result["left_pupil"] is None or result["right_pupil"] is None:
            st.warning(
                "Couldn't reliably locate the pupil in one or both eyes — try a "
                "closer, better-lit, sharper photo with eyes fully open."
            )
        else:
            left_op = result["left_opacity"]
            right_op = result["right_opacity"]
            sym = result["symmetry"]

            st.markdown("**Cataract opacity indicator**")
            c1, c2 = st.columns(2)
            c1.metric("Left eye", f"{left_op['opacity_score']:.0f}/100" if left_op else "N/A")
            c2.metric("Right eye", f"{right_op['opacity_score']:.0f}/100" if right_op else "N/A")
            st.caption("Higher = more visible clouding detected. Only flags advanced, visible clouding — see limitations above.")

            st.markdown("---")
            st.markdown("**Pupil symmetry indicator**")
            if sym:
                st.metric("Size difference between eyes", f"{sym['symmetry_diff_pct']:.1f}%")
                if sym["symmetry_diff_pct"] < 15:
                    st.caption("Within the range commonly seen from normal photo/lighting variation.")
                else:
                    st.caption(
                        "Noticeably different pupil sizes detected. This can be normal "
                        "physiological anisocoria (common, harmless) or a lighting "
                        "artifact — it is not, by itself, a diagnosis of anything."
                    )
            else:
                st.info("Couldn't compute symmetry — pupil not found in one or both eyes.")

    st.markdown("---")
    st.caption(
        "Reminder: both indicators are coarse, photo-quality-dependent screening aids "
        "for educational exploration — not a substitute for a real eye exam."
    )


if __name__ == "__main__":
    main()
