"""
test_pipeline.py
==================
Validates the MEASUREMENT math (pupil finding, opacity scoring,
symmetry scoring) using synthetic eye images with known, controlled
properties -- independent of Haar cascade face/eye detection, which
(as seen in the rPPG project) can behave differently across
environments and doesn't reliably fire on synthetic drawn images
anyway. This mirrors the same eyes-open validation approach used in
the rPPG and voice-biomarker projects: prove the math is right on
ground-truth synthetic data before it ever touches a real photo.
"""

import numpy as np
import cv2
from eye_analysis import find_pupil, compute_opacity_score, compute_symmetry

EYE_W, EYE_H = 200, 120


def make_synthetic_eye(pupil_radius, pupil_color_bgr, iris_color_bgr=(150, 100, 50),
                        sclera_color_bgr=(235, 235, 235), width=EYE_W, height=EYE_H):
    """Build a synthetic eye image: sclera background, iris ring, pupil circle."""
    canvas = np.full((height, width, 3), sclera_color_bgr, dtype=np.uint8)
    center = (width // 2, height // 2)
    iris_radius = int(pupil_radius * 2.2)
    cv2.circle(canvas, center, iris_radius, iris_color_bgr, -1)
    cv2.circle(canvas, center, pupil_radius, pupil_color_bgr, -1)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas, center


def test_pupil_detection():
    print("=== Pupil detection test ===")
    true_radius = 18
    healthy_dark = (8, 8, 8)  # near-black BGR
    eye_img, true_center = make_synthetic_eye(true_radius, healthy_dark)

    pupil = find_pupil(eye_img)
    assert pupil is not None, "Should detect a pupil in a clean synthetic eye image"
    cx, cy, r = pupil
    center_error = np.hypot(cx - true_center[0], cy - true_center[1])
    radius_error = abs(r - true_radius)

    print(f"  True center: {true_center}, detected: ({cx}, {cy}), center error: {center_error:.1f}px")
    print(f"  True radius: {true_radius}, detected: {r}, radius error: {radius_error}px")

    assert center_error < 8, f"Detected pupil center too far off ({center_error:.1f}px)"
    assert radius_error < 8, f"Detected pupil radius too far off ({radius_error}px)"
    print("  PASSED\n")
    return pupil, eye_img


def test_opacity_score_direction():
    print("=== Opacity score direction test ===")
    healthy_dark = (8, 8, 8)
    mild_cloud = (90, 90, 90)
    advanced_cloud = (210, 210, 215)  # whitish, low saturation -- classic leukocoria appearance

    healthy_img, _ = make_synthetic_eye(18, healthy_dark)
    mild_img, _ = make_synthetic_eye(18, mild_cloud)
    advanced_img, _ = make_synthetic_eye(18, advanced_cloud)

    healthy_pupil = find_pupil(healthy_img)
    mild_pupil = find_pupil(mild_img)
    advanced_pupil = find_pupil(advanced_img)

    assert healthy_pupil is not None and mild_pupil is not None and advanced_pupil is not None

    healthy_score = compute_opacity_score(healthy_img, healthy_pupil)["opacity_score"]
    mild_score = compute_opacity_score(mild_img, mild_pupil)["opacity_score"]
    advanced_score = compute_opacity_score(advanced_img, advanced_pupil)["opacity_score"]

    print(f"  Healthy (near-black) pupil opacity score: {healthy_score:.1f}")
    print(f"  Mild clouding pupil opacity score: {mild_score:.1f}")
    print(f"  Advanced clouding pupil opacity score: {advanced_score:.1f}")

    assert healthy_score < mild_score < advanced_score, (
        f"Opacity score should increase with injected clouding: {healthy_score:.1f}, {mild_score:.1f}, {advanced_score:.1f}"
    )
    assert healthy_score < 25, f"Healthy dark pupil should score low ({healthy_score:.1f})"
    assert advanced_score > 55, f"Advanced clouding should score clearly high ({advanced_score:.1f})"
    print("  PASSED\n")


def test_symmetry_score():
    print("=== Pupil symmetry test ===")
    left_radius = 15
    right_radius = 21  # deliberately 40% larger than left
    healthy_color = (10, 10, 10)

    left_img, _ = make_synthetic_eye(left_radius, healthy_color)
    right_img, _ = make_synthetic_eye(right_radius, healthy_color)

    left_pupil = find_pupil(left_img)
    right_pupil = find_pupil(right_img)
    assert left_pupil is not None and right_pupil is not None

    # Same eye bbox width for both (equal-scale synthetic images) -> symmetry
    # diff should closely track the actual detected radius ratio.
    left_bbox = (0, 0, EYE_W, EYE_H)
    right_bbox = (0, 0, EYE_W, EYE_H)

    result = compute_symmetry(left_pupil, left_bbox, right_pupil, right_bbox)
    assert result is not None

    detected_left_r = left_pupil[2]
    detected_right_r = right_pupil[2]
    expected_diff_pct = abs(detected_left_r - detected_right_r) / ((detected_left_r + detected_right_r) / 2) * 100

    print(f"  Injected radii: left={left_radius}, right={right_radius} (40% difference)")
    print(f"  Detected radii: left={detected_left_r}, right={detected_right_r}")
    print(f"  Computed symmetry diff: {result['symmetry_diff_pct']:.1f}%, expected (from detected radii): {expected_diff_pct:.1f}%")

    assert abs(result["symmetry_diff_pct"] - expected_diff_pct) < 1.0, "Symmetry calc should exactly match detected-radius ratio"
    assert result["symmetry_diff_pct"] > 15, "A 40% radius difference should register as clearly asymmetric"

    # Control case: equal pupils should show near-zero symmetry difference
    equal_img_a, _ = make_synthetic_eye(18, healthy_color)
    equal_img_b, _ = make_synthetic_eye(18, healthy_color)
    pupil_a = find_pupil(equal_img_a)
    pupil_b = find_pupil(equal_img_b)
    equal_result = compute_symmetry(pupil_a, left_bbox, pupil_b, right_bbox)
    print(f"  Equal pupils control case symmetry diff: {equal_result['symmetry_diff_pct']:.1f}%")
    assert equal_result["symmetry_diff_pct"] < 10, "Equal-size pupils should show low symmetry difference"

    print("  PASSED\n")


if __name__ == "__main__":
    test_pupil_detection()
    test_opacity_score_direction()
    test_symmetry_score()
    print("ALL PIPELINE CHECKS PASSED")
