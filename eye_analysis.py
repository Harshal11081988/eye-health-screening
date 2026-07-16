"""
eye_analysis.py
=================
Two photo-based screening indicators, computed from a normal frontal
face photo showing both eyes:

1. Cataract opacity indicator -- flags visible pupil clouding/whiteness,
   which is how ADVANCED cataracts actually present in an ordinary
   photo (leukocoria). This CANNOT detect early-stage cataracts,
   which require a slit-lamp exam to see.

2. Pupil symmetry (anisocoria) indicator -- flags a size difference
   between left and right pupils, a real neuro-ophthalmological
   screening sign, though ~20% of the population has mild, harmless
   physiological anisocoria.

Both are photo-quality-dependent, coarse indicators, not diagnoses.
See README for full limitations. Explicitly NOT built: any myopia/
hyperopia/prescription estimate -- refractive error cannot be derived
from a photo at all (see project conversation for why), so this
module makes no attempt at it.

Detection (face/eye finding) and measurement (pupil finding + scoring
on an already-cropped eye image) are kept as separate functions
deliberately, so the measurement math can be unit-tested on synthetic
eye images without depending on Haar cascade success -- see
test_pipeline.py.
"""

import numpy as np
import cv2

FACE_CASCADE_PATH_CANDIDATES = ["haarcascade_frontalface_default.xml"]
EYE_CASCADE_PATH_CANDIDATES = ["haarcascade_eye.xml"]


def _find_cascade_path(filename):
    """Robust cascade lookup -- same pattern used in the rPPG project
    after a real Streamlit Cloud failure with cv2.data auto-exposure."""
    candidates = []
    try:
        import cv2.data as cv2_data
        candidates.append(cv2_data.haarcascades + filename)
    except Exception:
        pass
    try:
        import os
        cv2_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        candidates.append(os.path.join(cv2_dir, "data", filename))
    except Exception:
        pass
    import os
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise RuntimeError(f"Could not locate cascade file: {filename}")


def detect_face_and_eyes(image_bgr):
    """
    Detect a face and, within it, the left/right eye bounding boxes.

    Returns a dict with 'face_bbox', 'left_eye_bbox', 'right_eye_bbox'
    (each as (x, y, w, h) in full-image coordinates), or raises
    ValueError with a clear message if detection fails.
    """
    face_cascade = cv2.CascadeClassifier(_find_cascade_path("haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(_find_cascade_path("haarcascade_eye.xml"))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        raise ValueError("No face detected. Use a clear, well-lit, front-facing photo.")

    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face
    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

    # Restrict eye search to the upper ~60% of the face (avoids nostril/mouth false positives)
    upper_face_gray = face_roi_gray[0:int(fh * 0.6), :]
    eyes = eye_cascade.detectMultiScale(upper_face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))

    if len(eyes) < 2:
        raise ValueError(
            f"Could only detect {len(eyes)} eye(s) — need both eyes clearly visible and open. "
            "Try a straight-on photo with even lighting."
        )

    # Keep the two largest detections (most likely to be real eyes, not eyebrows/glasses glare)
    eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    # Distinguish left/right by x-position (in the image, the person's right eye appears on the left side)
    eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])
    image_left_eye, image_right_eye = eyes_sorted[0], eyes_sorted[1]

    def to_full_coords(eye_bbox):
        ex, ey, ew, eh = eye_bbox
        return (fx + ex, fy + ey, ew, eh)

    return {
        "face_bbox": (fx, fy, fw, fh),
        "left_eye_bbox": to_full_coords(image_left_eye),   # left side of the image
        "right_eye_bbox": to_full_coords(image_right_eye),  # right side of the image
    }


def find_pupil(eye_roi_bgr):
    """
    Locate the pupil within a cropped eye image via radial gradient
    scanning: cast many rays outward from an estimated center and find
    where each ray crosses from dark (pupil) to lighter (iris), then
    take the median crossing distance as the pupil radius.

    This replaced an earlier threshold+contour-area-jump approach that
    testing showed was bistable near a sharp pupil/iris boundary --
    small pixel noise could flip a single hard threshold decision and
    make two genuinely IDENTICAL pupils register as different sizes
    (see test_pipeline.py history / README for the full story).
    Aggregating an edge estimate over ~70 independent angular samples
    (median, which is robust to a few noisy/occluded rays) is a
    standard, much more stable technique from iris/pupil segmentation
    literature.

    Returns (center_x, center_y, radius) in eye_roi coordinates, or
    None if no plausible pupil region was found.
    """
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]

    min_r = max(int(min(h, w) * 0.06), 3)
    max_r = int(min(h, w) * 0.45)

    # Step 1: seed center estimate via darkest-region centroid (a coarse
    # starting point; the radial scan below does the real precision work)
    flat_sorted = np.sort(gray.flatten())
    seed_thresh = max(int(flat_sorted[max(int(len(flat_sorted) * 0.1), 1)]), 5)
    _, seed_binary = cv2.threshold(gray, seed_thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(seed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = np.pi * (min_r ** 2)
    max_area = np.pi * (max_r ** 2)
    plausible = [c for c in contours if min_area * 0.3 <= cv2.contourArea(c) <= max_area * 1.5]
    if not plausible:
        plausible = contours

    def dist_to_center(c):
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        return np.hypot(cx - w / 2, cy - h / 2)

    seed_contour = min(plausible, key=dist_to_center)
    (seed_cx, seed_cy), _ = cv2.minEnclosingCircle(seed_contour)

    # Step 2: radial gradient scan -- find the dark-to-light edge along
    # many rays from the seed center, take the median distance
    n_rays = 72
    edge_radii = []

    for i in range(n_rays):
        theta = 2 * np.pi * i / n_rays
        dx, dy = np.cos(theta), np.sin(theta)

        profile = []
        for r in range(1, max_r + 5):
            x = int(round(seed_cx + dx * r))
            y = int(round(seed_cy + dy * r))
            if 0 <= x < w and 0 <= y < h:
                profile.append(gray[y, x])
            else:
                break

        if len(profile) < min_r + 3:
            continue

        profile = np.array(profile, dtype=np.float64)
        grad = np.diff(profile)

        search_end = min(len(grad), max_r)
        if search_end <= min_r:
            continue

        # Find the FIRST meaningfully strong dark-to-light edge scanning
        # outward from the center, not the single strongest edge overall.
        # The pupil/iris boundary is the innermost such edge; the
        # iris/sclera boundary further out is usually a stronger edge
        # in absolute terms, and testing showed searching for the global
        # max picks that outer boundary instead of the pupil.
        edge_threshold = 3.0
        found_r = None
        for r_idx in range(min_r, search_end):
            if grad[r_idx] >= edge_threshold:
                found_r = r_idx
                break

        if found_r is None:
            continue

        edge_radii.append(found_r)

    if len(edge_radii) < n_rays * 0.3:
        return None  # not enough consistent edges found -- unreliable image

    final_r = int(round(float(np.median(edge_radii))))
    return (int(round(seed_cx)), int(round(seed_cy)), final_r)


def compute_opacity_score(eye_roi_bgr, pupil_circle):
    """
    Cataract opacity indicator: how bright/whitish the pupil region is.
    A healthy pupil is very dark (near-black) in a normal photo;
    visible clouding raises brightness and lowers color saturation.

    Returns a 0-100 score (higher = more visible clouding detected)
    plus the raw brightness/saturation values, or None if no pupil found.
    """
    if pupil_circle is None:
        return None

    cx, cy, r = pupil_circle
    hsv = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(eye_roi_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), max(r - 1, 1), 255, -1)

    mean_v = cv2.mean(hsv[:, :, 2], mask=mask)[0]  # brightness/value
    mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]  # saturation

    # Healthy pupil: low V (dark), saturation is largely irrelevant when
    # V is already low. Clouding: V rises AND saturation tends to drop
    # (grayish-white rather than colored). Combine into one 0-100 score.
    brightness_component = np.clip(mean_v / 255.0, 0, 1)
    low_saturation_component = np.clip(1 - (mean_s / 255.0), 0, 1)
    opacity_score = float(np.clip(100 * (0.7 * brightness_component + 0.3 * low_saturation_component), 0, 100))

    return {
        "opacity_score": opacity_score,
        "mean_brightness": float(mean_v),
        "mean_saturation": float(mean_s),
    }


def compute_symmetry(left_pupil_circle, left_eye_bbox, right_pupil_circle, right_eye_bbox):
    """
    Pupil symmetry indicator: compares pupil size between the two eyes,
    normalized by each eye's detected bounding-box width (a stable
    scale reference that doesn't require accurate iris detection).

    Returns a dict with normalized sizes and a percent-difference
    symmetry score, or None if either pupil wasn't found.
    """
    if left_pupil_circle is None or right_pupil_circle is None:
        return None

    left_w = left_eye_bbox[2]
    right_w = right_eye_bbox[2]

    left_norm = left_pupil_circle[2] / left_w
    right_norm = right_pupil_circle[2] / right_w

    if left_norm <= 0 or right_norm <= 0:
        return None

    avg_norm = (left_norm + right_norm) / 2.0
    diff_pct = float(abs(left_norm - right_norm) / avg_norm * 100.0)

    return {
        "left_normalized_size": float(left_norm),
        "right_normalized_size": float(right_norm),
        "symmetry_diff_pct": diff_pct,
    }


def analyze_eye_photo(image_bgr):
    """
    Full pipeline: detect face/eyes, find pupils, compute both
    indicators. Returns a summary dict. Raises ValueError with a
    user-facing message if detection fails at any required step.
    """
    detection = detect_face_and_eyes(image_bgr)

    lx, ly, lw, lh = detection["left_eye_bbox"]
    rx, ry, rw, rh = detection["right_eye_bbox"]
    left_eye_roi = image_bgr[ly:ly + lh, lx:lx + lw]
    right_eye_roi = image_bgr[ry:ry + rh, rx:rx + rw]

    left_pupil = find_pupil(left_eye_roi)
    right_pupil = find_pupil(right_eye_roi)

    left_opacity = compute_opacity_score(left_eye_roi, left_pupil)
    right_opacity = compute_opacity_score(right_eye_roi, right_pupil)
    symmetry = compute_symmetry(left_pupil, detection["left_eye_bbox"], right_pupil, detection["right_eye_bbox"])

    return {
        "detection": detection,
        "left_eye_roi": left_eye_roi,
        "right_eye_roi": right_eye_roi,
        "left_pupil": left_pupil,
        "right_pupil": right_pupil,
        "left_opacity": left_opacity,
        "right_opacity": right_opacity,
        "symmetry": symmetry,
    }
