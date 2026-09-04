"""
Phase 4: Face Detection & Video Frame Extraction Module
AI-Powered Deepfake Detection System

This module handles:
1. Multi-cascade face detection with automatic XML download fallback.
2. Square aspect-ratio preserved face cropping with margin padding to prevent facial distortion.
3. Bounding box visual annotation overlay.
4. Video processing: extracting sequential frames from input videos (.mp4, .avi, etc.).
"""

import os
import urllib.request
import cv2
import numpy as np
from PIL import Image

DEFAULT_TARGET_SIZE = (128, 128)
CASCADE_DIR = "models"
PRIMARY_CASCADE = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
ALT_CASCADE = os.path.join(CASCADE_DIR, "haarcascade_frontalface_alt2.xml")
PROFILE_CASCADE = os.path.join(CASCADE_DIR, "haarcascade_profileface.xml")

CASCADE_URLS = {
    PRIMARY_CASCADE: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    ALT_CASCADE: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",
    PROFILE_CASCADE: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml",
}

def ensure_cascade_files():
    """
    Ensures OpenCV Haar Cascade XML files exist locally. Downloads missing XML files automatically.
    """
    os.makedirs(CASCADE_DIR, exist_ok=True)
    for path, url in CASCADE_URLS.items():
        if not os.path.exists(path):
            # Check cv2 internal data first
            cv2_internal = os.path.join(getattr(cv2.data, 'haarcascades', ''), os.path.basename(path))
            if os.path.exists(cv2_internal):
                continue
            try:
                print(f"[Face Extractor] Downloading cascade: {os.path.basename(path)}...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  - Warning: Could not download {path} ({e})")

def load_cascades():
    """
    Loads list of available OpenCV Cascade Classifiers in order of priority.
    """
    ensure_cascade_files()
    cascades = []
    
    # Try cv2 built-in paths first
    cv2_data = getattr(cv2.data, 'haarcascades', '')
    paths = [
        os.path.join(cv2_data, 'haarcascade_frontalface_default.xml'),
        os.path.join(cv2_data, 'haarcascade_frontalface_alt2.xml'),
        os.path.join(cv2_data, 'haarcascade_profileface.xml'),
        PRIMARY_CASCADE,
        ALT_CASCADE,
        PROFILE_CASCADE
    ]
    
    loaded_paths = set()
    for p in paths:
        if os.path.exists(p) and p not in loaded_paths:
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                cascades.append(c)
                loaded_paths.add(p)
                
    return cascades

def draw_face_bounding_box(image_rgb, bbox, label="Face Detected", color=(0, 242, 254)):
    """
    Draws a modern, stylized bounding box with corner accents on the RGB image.
    """
    annotated = image_rgb.copy()
    if bbox is None:
        return annotated
        
    x, y, w, h = bbox
    img_h, img_w = annotated.shape[:2]
    
    # Convert RGB color to BGR for OpenCV drawing
    r, g, b = color
    bgr_color = (b, g, r)
    
    # Draw outer bounding box rectangle
    thickness = max(2, int(min(img_h, img_w) / 150))
    cv2.rectangle(annotated, (x, y), (x + w, y + h), bgr_color, thickness)
    
    # Draw corner accent accents
    line_len = max(10, int(min(w, h) * 0.15))
    accent_thick = thickness + 2
    
    # Top-Left
    cv2.line(annotated, (x, y), (x + line_len, y), bgr_color, accent_thick)
    cv2.line(annotated, (x, y), (x, y + line_len), bgr_color, accent_thick)
    # Top-Right
    cv2.line(annotated, (x + w, y), (x + w - line_len, y), bgr_color, accent_thick)
    cv2.line(annotated, (x + w, y), (x + w, y + line_len), bgr_color, accent_thick)
    # Bottom-Left
    cv2.line(annotated, (x, y + h), (x + line_len, y + h), bgr_color, accent_thick)
    cv2.line(annotated, (x, y + h), (x, y + h - line_len), bgr_color, accent_thick)
    # Bottom-Right
    cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), bgr_color, accent_thick)
    cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), bgr_color, accent_thick)
    
    # Draw Label Tag Background
    font_scale = max(0.4, min(img_h, img_w) / 600.0)
    font_thick = max(1, int(font_scale * 1.8))
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    
    tag_y1 = max(0, y - text_h - 10)
    tag_y2 = max(text_h + 10, y)
    cv2.rectangle(annotated, (x, tag_y1), (x + text_w + 14, tag_y2), bgr_color, -1)
    cv2.putText(annotated, label, (x + 7, tag_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)
    
    return annotated

def extract_face_from_image(image_input, target_size=DEFAULT_TARGET_SIZE, margin_ratio=0.35):
    """
    Detects and crops the primary human face from an input image using square padding margin.
    
    Args:
        image_input (str, PIL.Image, or np.ndarray): Input image.
        target_size (tuple): Resized output dimensions (default 128x128).
        margin_ratio (float): Padding margin around face box (default 0.35 = 35%).
        
    Returns:
        tuple: (cropped_face_rgb, face_detected_bool, bounding_box_tuple, annotated_image_rgb)
    """
    img_bgr = None
    
    if hasattr(image_input, "convert"):
        # PIL Image
        pil_img = image_input.convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        # NumPy array — check if it's RGB or BGR
        if image_input.ndim == 2:
            # Grayscale — convert to BGR
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            # RGBA — drop alpha, treat as RGB then convert
            img_bgr = cv2.cvtColor(image_input[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            # Assume RGB numpy (from PIL/st.camera_input)
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            try:
                pil_img = Image.open(image_input).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Unable to read image file: {image_input}")
    else:
        try:
            pil_img = Image.open(image_input).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
    img_h, img_w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram for better detection in dark/uneven images
    gray = cv2.equalizeHist(gray)
    
    cascades = load_cascades()
    detected_rect = None
    
    # Try detection across cascades and scale factors
    for cascade in cascades:
        for scale in [1.05, 1.1, 1.15]:
            for min_neighbors in [3, 4, 5]:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=min_neighbors,
                    minSize=(20, 20)
                )
                if len(faces) > 0:
                    # Pick largest face by area
                    detected_rect = max(faces, key=lambda rect: rect[2] * rect[3])
                    break
            if detected_rect is not None:
                break
        if detected_rect is not None:
            break
            
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if detected_rect is None:
        # No face detected — use center crop fallback
        shorter = min(img_h, img_w)
        cx = img_w // 2
        cy = img_h // 2
        half = shorter // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img_w, cx + half)
        y2 = min(img_h, cy + half)
        center_crop = img_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(center_crop, target_size)
        fallback_bbox = (x1, y1, x2 - x1, y2 - y1)
        annotated_img = draw_face_bounding_box(img_rgb, fallback_bbox, label="CENTER CROP (NO FACE)", color=(255, 170, 0))
        return face_resized, False, fallback_bbox, annotated_img

    # Face was detected
    x, y, w, h = detected_rect
    raw_bbox = (int(x), int(y), int(w), int(h))
    
    # Calculate face center
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    
    # Determine square side size with margin padding
    max_dim = max(w, h)
    square_size = max_dim * (1.0 + margin_ratio)
    
    # Compute square boundaries
    x1 = int(round(center_x - square_size / 2.0))
    y1 = int(round(center_y - square_size / 2.0))
    x2 = int(round(center_x + square_size / 2.0))
    y2 = int(round(center_y + square_size / 2.0))
    
    # Compute padding needed if square extends beyond image dimensions
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img_w)
    
    # Clamp crop coordinates to original image size
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)
    
    face_crop_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Pad crop if necessary to guarantee exact square shape without stretching
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        face_crop_bgr = cv2.copyMakeBorder(
            face_crop_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT
        )
        
    # Convert BGR crop to RGB
    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_crop_rgb, target_size)
    
    # Create annotated RGB image with bounding box overlay
    annotated_img = draw_face_bounding_box(img_rgb, raw_bbox, label="FACE DETECTED", color=(0, 242, 254))
    
    return face_resized, True, raw_bbox, annotated_img

def extract_frames_from_video(video_path, max_frames=10, frame_interval=15, target_size=DEFAULT_TARGET_SIZE):
    """
    Extracts face frames sequentially from a video file at regular frame intervals.
    
    Returns:
        tuple: (extracted_face_tensors, total_frames, faces_detected_count, keyframe_dicts)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV failed to open video stream: {video_path}")
        
    extracted_face_frames = []
    keyframe_details = []
    frame_count = 0
    faces_detected_count = 0
    
    while cap.isOpened() and len(extracted_face_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            # frame from cv2 is BGR — pass directly as numpy BGR array
            face_img, face_found, bbox, annotated = extract_face_from_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                target_size=target_size
            )
            # Apply MobileNetV2 preprocessing
            import tensorflow as tf
            face_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
                face_img.astype(np.float32)
            )
            extracted_face_frames.append(face_tensor)
            
            if face_found:
                faces_detected_count += 1
                
            keyframe_details.append({
                "frame_num": frame_count,
                "face_rgb": face_img,
                "annotated_rgb": annotated,
                "face_found": face_found,
                "bbox": bbox
            })
            
    cap.release()
    return extracted_face_frames, frame_count, faces_detected_count, keyframe_details

def create_sample_test_video(output_video_path="dataset/sample_test_video.mp4", duration_sec=2, fps=15):
    """
    Generates a small synthetic test video with an animated face graphic for testing.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (256, 256))
    
    total_frames = duration_sec * fps
    for i in range(total_frames):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :] = (180, 150, 120)
        
        x_center = 80 + int(i * 3)
        cv2.circle(frame, (x_center, 128), 50, (220, 200, 180), -1)
        cv2.circle(frame, (x_center - 15, 110), 6, (50, 30, 20), -1)
        cv2.circle(frame, (x_center + 15, 110), 6, (50, 30, 20), -1)
        cv2.ellipse(frame, (x_center, 145), (20, 10), 0, 0, 180, (50, 30, 20), 2)
        
        out.write(frame)
        
    out.release()


if __name__ == "__main__":
    print("Face extractor module loaded OK.")
