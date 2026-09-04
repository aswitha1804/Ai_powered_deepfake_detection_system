"""
Dataset Generator for Deepfake Detection System
Generates synthetic REAL and FAKE face images with distinguishable features.

REAL faces:
- Natural skin tone with slight natural texture variation
- Symmetric eyes at a natural position
- Subtle natural noise

FAKE faces (simulated deepfake artifacts):
- Hue/color shift (color bleed at boundaries)
- Eye asymmetry (misaligned, different sizes)
- Boundary blending seams (horizontal/vertical gradient)
- Over-smoothing blur artifacts
- Unnatural skin tone inconsistency patches
"""

import os
import cv2
import numpy as np
import random

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "dataset"
IMG_SIZE = 128
NUM_TRAIN_REAL = 400
NUM_TRAIN_FAKE = 400
NUM_VAL_REAL = 100
NUM_VAL_FAKE = 100
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


def draw_real_face(img_size=128):
    """
    Draws a synthetic REAL human face with natural features.
    Returns: RGB numpy array (img_size, img_size, 3)
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Background — natural indoor lighting
    bg_color = (
        random.randint(180, 230),
        random.randint(160, 210),
        random.randint(140, 190)
    )
    img[:, :] = bg_color

    # Skin tone — warm natural range
    skin_r = random.randint(190, 240)
    skin_g = random.randint(140, 185)
    skin_b = random.randint(110, 155)

    cx, cy = img_size // 2, img_size // 2
    face_rx = random.randint(30, 40)
    face_ry = random.randint(35, 46)

    # Face ellipse
    cv2.ellipse(img, (cx, cy), (face_rx, face_ry), 0, 0, 360, (skin_b, skin_g, skin_r), -1)

    # Add subtle natural skin texture noise
    noise = np.random.randint(-8, 8, (img_size, img_size, 3), dtype=np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Keep only face region with noise
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (face_rx, face_ry), 0, 0, 360, 255, -1)
    img = np.where(mask[:, :, np.newaxis] > 0, noisy_img, img)

    # Eyes — symmetric, at natural position
    eye_y = cy - face_ry // 4
    eye_spacing = face_rx // 2
    eye_size = random.randint(5, 7)

    # Left eye
    lx = cx - eye_spacing
    cv2.ellipse(img, (lx, eye_y), (eye_size, eye_size - 1), 0, 0, 360, (40, 25, 15), -1)
    cv2.ellipse(img, (lx, eye_y), (eye_size - 2, eye_size - 3), 0, 0, 360, (10, 10, 40), -1)
    cv2.circle(img, (lx + 1, eye_y - 1), 1, (200, 200, 220), -1)  # highlight

    # Right eye — same size (symmetric)
    rx_e = cx + eye_spacing
    cv2.ellipse(img, (rx_e, eye_y), (eye_size, eye_size - 1), 0, 0, 360, (40, 25, 15), -1)
    cv2.ellipse(img, (rx_e, eye_y), (eye_size - 2, eye_size - 3), 0, 0, 360, (10, 10, 40), -1)
    cv2.circle(img, (rx_e + 1, eye_y - 1), 1, (200, 200, 220), -1)

    # Nose — subtle
    nose_y = cy + face_ry // 8
    cv2.ellipse(img, (cx, nose_y), (5, 3), 0, 0, 360, (skin_b - 20, skin_g - 20, skin_r - 20), -1)

    # Mouth — natural symmetric smile
    mouth_y = cy + face_ry // 2 - 5
    cv2.ellipse(img, (cx, mouth_y), (face_rx // 3, 5), 0, 0, 180, (100, 60, 60), 2)

    # Eyebrows — symmetric
    brow_y = eye_y - eye_size - 3
    brow_w = eye_size + 3
    cv2.line(img, (lx - brow_w, brow_y + 2), (lx + brow_w, brow_y - 2), (50, 30, 20), 2)
    cv2.line(img, (rx_e - brow_w, brow_y - 2), (rx_e + brow_w, brow_y + 2), (50, 30, 20), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_fake_face(img_size=128):
    """
    Draws a synthetic FAKE (deepfake) face with visible artifacts.
    Returns: RGB numpy array (img_size, img_size, 3)
    """
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Background
    bg_color = (
        random.randint(160, 220),
        random.randint(140, 200),
        random.randint(120, 180)
    )
    img[:, :] = bg_color

    # Skin tone — slightly off/unnatural hue shift
    skin_r = random.randint(190, 240)
    skin_g = random.randint(140, 185)
    skin_b = random.randint(110, 155)

    cx, cy = img_size // 2, img_size // 2
    face_rx = random.randint(30, 40)
    face_ry = random.randint(35, 46)

    # ARTIFACT 1: Unnatural color patch bleed
    # The face color differs from surrounding area with a visible seam
    hue_shift = random.randint(30, 60)
    fake_skin_b = min(255, skin_b + hue_shift)
    cv2.ellipse(img, (cx, cy), (face_rx, face_ry), 0, 0, 360, (fake_skin_b, skin_g, skin_r), -1)

    # ARTIFACT 2: Blending seam — unnatural horizontal band
    seam_y = cy + random.randint(-5, 5)
    seam_color = (
        random.randint(90, 140),
        random.randint(60, 110),
        random.randint(50, 100)
    )
    cv2.line(img, (cx - face_rx, seam_y), (cx + face_rx, seam_y), seam_color, random.randint(2, 4))

    # ARTIFACT 3: Over-smoothing (strong Gaussian blur on face region)
    face_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.ellipse(face_mask, (cx, cy), (face_rx, face_ry), 0, 0, 360, 255, -1)
    blurred = cv2.GaussianBlur(img, (11, 11), 5)
    img = np.where(face_mask[:, :, np.newaxis] > 0, blurred, img)

    # ARTIFACT 4: Asymmetric eyes — different sizes and slight vertical offset
    eye_y = cy - face_ry // 4
    eye_spacing = face_rx // 2

    # Left eye (larger)
    lx = cx - eye_spacing
    l_eye_size = random.randint(6, 9)
    cv2.ellipse(img, (lx, eye_y), (l_eye_size, l_eye_size - 1), 0, 0, 360, (40, 25, 15), -1)
    cv2.ellipse(img, (lx, eye_y), (l_eye_size - 2, l_eye_size - 3), 0, 0, 360, (10, 10, 40), -1)

    # Right eye (noticeably different size + vertical offset = asymmetry artifact)
    rx_e = cx + eye_spacing
    r_eye_size = max(5, l_eye_size - random.randint(2, 4))  # smaller right eye, min 5
    r_eye_y = eye_y + random.randint(3, 7)  # vertically misaligned
    cv2.ellipse(img, (rx_e, r_eye_y), (r_eye_size, max(1, r_eye_size - 1)), 0, 0, 360, (40, 25, 15), -1)
    cv2.ellipse(img, (rx_e, r_eye_y), (max(1, r_eye_size - 2), max(1, r_eye_size - 3)), 0, 0, 360, (10, 10, 40), -1)

    # Nose
    nose_y = cy + face_ry // 8
    cv2.ellipse(img, (cx, nose_y), (5, 3), 0, 0, 360, (fake_skin_b - 20, skin_g - 20, skin_r - 20), -1)

    # Mouth — slightly asymmetric
    mouth_y = cy + face_ry // 2 - 5
    mouth_offset = random.randint(-5, 5)
    cv2.ellipse(img, (cx + mouth_offset, mouth_y), (face_rx // 3, 5), 0, 0, 180, (100, 60, 60), 2)

    # ARTIFACT 5: Unnatural skin inconsistency patches
    for _ in range(random.randint(3, 6)):
        px = cx + random.randint(-face_rx + 5, face_rx - 5)
        py = cy + random.randint(-face_ry + 5, face_ry - 5)
        patch_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
        patch_r = random.randint(2, 6)
        cv2.circle(img, (px, py), patch_r, patch_color, -1)

    # Apply slight additional blur to entire image (over-processed look)
    img = cv2.GaussianBlur(img, (5, 5), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def generate_dataset(
    output_dir=OUTPUT_DIR,
    img_size=IMG_SIZE,
    num_train_real=NUM_TRAIN_REAL,
    num_train_fake=NUM_TRAIN_FAKE,
    num_val_real=NUM_VAL_REAL,
    num_val_fake=NUM_VAL_FAKE,
):
    """
    Generates and saves synthetic dataset images to disk.
    """
    splits = {
        "train/real":  (num_train_real, draw_real_face),
        "train/fake":  (num_train_fake, draw_fake_face),
        "validation/real": (num_val_real, draw_real_face),
        "validation/fake": (num_val_fake, draw_fake_face),
    }

    total = 0
    for split_path, (count, draw_fn) in splits.items():
        save_dir = os.path.join(output_dir, split_path)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {count} images -> {save_dir}")
        for i in range(count):
            img_rgb = draw_fn(img_size)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            filename = os.path.join(save_dir, f"{i:04d}.jpg")
            cv2.imwrite(filename, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1

    print(f"\nDataset generation complete! {total} images saved to '{output_dir}/'")
    return total


if __name__ == "__main__":
    print("=" * 60)
    print("  Deepfake Detection - Synthetic Dataset Generator")
    print("=" * 60)
    generate_dataset()
