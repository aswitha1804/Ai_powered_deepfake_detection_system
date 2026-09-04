"""
Phase 10: Automated Unit & Integration Testing Suite
AI-Powered Deepfake Detection System

This test suite verifies:
1. Environment library availability.
2. Dataset directory hierarchy integrity.
3. Preprocessing image shape and normalization bounds.
4. Face extraction bounding box generation.
5. CNN model architecture shape and forward pass compilation.
6. Prediction engine inference dictionary structure.
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Add project root directory to Python path for reliable module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_and_preprocess_image, create_dataset_pipelines
from src.face_extractor import extract_face_from_image, extract_frames_from_video
from src.model import build_deepfake_detection_model
from src.predict import predict_image, predict_video

class TestDeepfakeDetectionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test file paths before running tests."""
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.dataset_dir = os.path.join(cls.project_root, "dataset")
        cls.real_sample_dir = os.path.join(cls.dataset_dir, "train", "real")
        cls.fake_sample_dir = os.path.join(cls.dataset_dir, "train", "fake")
        cls.video_sample_path = os.path.join(cls.dataset_dir, "sample_test_video.mp4")
        cls.model_path = os.path.join(cls.project_root, "models", "best_deepfake_model.keras")

    def test_01_environment_imports(self):
        """Test that all required libraries can be imported cleanly."""
        import cv2
        import pandas
        import matplotlib
        import sklearn
        import streamlit
        self.assertIsNotNone(cv2.__version__)
        self.assertIsNotNone(tf.__version__)

    def test_02_dataset_directory_structure(self):
        """Test that dataset subfolders exist and contain image files."""
        self.assertTrue(os.path.exists(self.real_sample_dir))
        self.assertTrue(os.path.exists(self.fake_sample_dir))
        
        real_files = [f for f in os.listdir(self.real_sample_dir) if f.endswith(('.jpg', '.png'))]
        fake_files = [f for f in os.listdir(self.fake_sample_dir) if f.endswith(('.jpg', '.png'))]
        
        self.assertGreater(len(real_files), 0, "No sample real images found!")
        self.assertGreater(len(fake_files), 0, "No sample fake images found!")

    def test_03_preprocessing_tensor_output(self):
        """Test image preprocessing output shape and normalization limits."""
        sample_img_path = os.path.join(self.real_sample_dir, os.listdir(self.real_sample_dir)[0])
        tensor = load_and_preprocess_image(sample_img_path, target_size=(128, 128))
        
        self.assertEqual(tensor.shape, (1, 128, 128, 3))
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)
        self.assertEqual(tensor.dtype, np.float32)

    def test_04_face_extractor(self):
        """Test face extraction on single image."""
        sample_img_path = os.path.join(self.real_sample_dir, os.listdir(self.real_sample_dir)[0])
        face_crop, face_found, bbox, annotated_img = extract_face_from_image(sample_img_path, target_size=(128, 128))
        
        self.assertEqual(face_crop.shape, (128, 128, 3))
        self.assertIsInstance(face_found, bool)
        self.assertEqual(len(bbox), 4)
        self.assertEqual(annotated_img.shape[2], 3)

    def test_05_model_architecture(self):
        """Test CNN model compilation and forward pass output shape."""
        model = build_deepfake_detection_model(input_shape=(128, 128, 3))
        dummy_input = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
        dummy_output = model(dummy_input)
        
        self.assertEqual(dummy_output.shape, (1, 1))
        self.assertGreaterEqual(float(dummy_output[0][0]), 0.0)
        self.assertLessEqual(float(dummy_output[0][0]), 1.0)

    def test_06_prediction_engine_image(self):
        """Test prediction engine output dictionary for images."""
        if os.path.exists(self.model_path):
            sample_img_path = os.path.join(self.real_sample_dir, os.listdir(self.real_sample_dir)[0])
            result = predict_image(sample_img_path, model_path=self.model_path)
            
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("raw_prob", result)
            self.assertIn(result["label"], ["REAL", "DEEPFAKE"])
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 100.0)

    def test_07_prediction_engine_video(self):
        """Test prediction engine output dictionary for videos."""
        if os.path.exists(self.model_path) and os.path.exists(self.video_sample_path):
            result = predict_video(self.video_sample_path, model_path=self.model_path)
            
            self.assertIn("label", result)
            self.assertIn("confidence", result)
            self.assertIn("frame_breakdown", result)
            self.assertIn(result["label"], ["REAL", "DEEPFAKE"])
            self.assertGreater(len(result["frame_breakdown"]), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
