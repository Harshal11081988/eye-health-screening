# 👁️ Eye Health Screening Indicators

Two photo-based screening indicators computed from a normal
front-facing photo:

1. **Cataract opacity indicator** — flags visible pupil clouding
2. **Pupil symmetry (anisocoria) indicator** — flags left/right pupil
   size differences

> ⚠️ **Not a medical device or diagnostic tool.** See "What this
> can't do" below — it's as important as what it can do.

## What this deliberately does NOT include, and why

**No glaucoma detection.** Real glaucoma diagnosis requires imaging
the optic nerve at the back of the eye (fundus camera or OCT scan) to
measure cupping/damage, plus intraocular pressure measurement
(tonometry). Neither is visible in an external photo — a phone camera
cannot see the structure glaucoma damages. Building a "glaucoma
detector" from a selfie would measure something with no real
connection to the disease. Left out entirely rather than faked.

**No myopia/hyperopia/prescription number.** A glasses prescription
(diopter power) is determined by either subjective refraction (an
optometrist testing lens options against your actual focusing eye) or
objective autorefraction (specialized hardware measuring infrared
light reflection inside the eye). There is no external visual feature
in any photo that correlates with refractive error — this isn't an
accuracy limitation like the ones below, it's a complete absence of
signal. A tool that output a fake diopter number could lead someone
to buy the wrong glasses. Not built, on principle.

## What the two included indicators can and can't do

**Cataract opacity indicator:**
- Only flags *visible* clouding — how *advanced* cataracts present in
  an ordinary photo (leukocoria)
- Cannot detect early-stage cataracts, which require a slit-lamp exam
- A low score means "no advanced visible clouding in this photo," not
  "healthy eyes"

**Pupil symmetry (anisocoria) indicator:**
- Compares left/right pupil size, normalized by detected eye-region
  width
- ~20% of people have mild, completely harmless physiological
  anisocoria — a detected difference alone isn't a red flag
- Lighting affects pupil size (constriction/dilation) — uneven
  lighting between eyes in one photo can create a false asymmetry
  reading unrelated to anything physiological

## How it was validated

Face/eye detection (via OpenCV Haar cascades — no external download,
same bundled-cascade approach as the rPPG project, including its
`cv2.data` failure-mode fix) is inherently hard to validate on
synthetic images, since Haar cascades are trained on real faces. So
detection and measurement were tested separately:

- **Measurement math** (pupil finding, opacity scoring, symmetry
  calculation) was validated on synthetic eye images with known,
  controlled properties — see `test_pipeline.py`.

- **Two real bugs were caught and fixed during this validation, not
  assumed away:**
  1. The first pupil-finding approach (Hough Circle Transform)
     consistently locked onto the iris boundary instead of the pupil
     boundary, even on a clean synthetic image — the iris/sclera edge
     has higher contrast than the pupil/iris edge, so Hough kept
     finding the wrong circle. Switched to threshold + contour
     analysis.
  2. That fix introduced a *second*, subtler bug: a hard
     threshold-crossing decision near a sharp boundary turned out to
     be bistable — re-running the exact same synthetic pupil twice
     (identical size, different random pixel noise) sometimes
     produced detected radii several pixels apart, which would show
     up as a false ~15-40% "asymmetry" between two genuinely identical
     eyes. Caught by a control-case test that measures two supposedly
     equal pupils and checks the reported difference stays near zero.
     Fixed by switching to radial gradient scanning (casting ~70 rays
     outward from the pupil center and taking the *median* dark-to-light
     edge crossing distance) — a standard, statistically robust
     technique from iris/pupil segmentation literature, rather than one
     brittle global threshold decision.
  - Final validated behavior: pupil center detection near-exact,
    radius detection consistent (0% variance across 15 repeated trials
    on identical synthetic pupils), symmetry calculation exactly
    matches detected-radius ratios, and opacity scores show correct
    monotonic response to injected clouding severity.

## Project structure

```
eye-health-screening/
├── app.py                 # Streamlit app (deploy this)
├── eye_analysis.py          # Core detection + measurement pipeline
├── test_pipeline.py           # Synthetic-image validation (see above)
├── requirements.txt
└── README.md
```

## Setup (local)

```bash
git clone <your-repo-url>
cd eye-health-screening
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python test_pipeline.py   # optional: verify the pipeline on your machine

streamlit run app.py
```

## Deploying to Streamlit Community Cloud

No dataset, no training step, no network dependency at runtime — push
the repo as-is and point Streamlit Cloud at `app.py`.

## Tech stack

- **OpenCV (headless)** — face/eye detection (Haar cascades), pupil
  detection (thresholding + contour analysis)
- **NumPy** — signal/array processing
- **Pillow** — image I/O
- **Streamlit** — UI
- **Plotly** — reserved for future trend visualizations

## Photo tips

- Even, bright lighting on both eyes equally (mismatched lighting
  between eyes is the most common cause of a false symmetry reading)
- Look straight at the camera, eyes fully open
- No glasses glare directly over the pupil

## Possible extensions

- Session-based tracking across multiple photos (same pattern as the
  Neuroplasticity Tracker and Voice Biomarker projects)
- Red-reflex analysis (a more clinically-grounded cataract screening
  technique, but requires a flash-photography protocol most phone
  cameras don't standardize)
- Iris color/texture analysis as an additional normalization signal
  instead of eye-bbox width
