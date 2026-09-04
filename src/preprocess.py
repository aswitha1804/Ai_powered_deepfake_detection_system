"""
Phase 3: Data Preprocessing & Data Pipeline Module
AI-Powered Deepfake Detection System

This module handles:
1. Single image loading, RGB conversion, resizing, and pixel normalization [0.0, 1.0].
2. Mini-batch dataset creation using TensorFlow for training and validation sets.
3. Data augmentation (rotations, flips, zoom) to reduce overfitting.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

DEFAULT_TARGET_SIZE = (128, 128)


def load_and_preprocess_image(image_path, target_size=DEFAULT_TARGET_SIZE):
    """
    Load one image and preprocess it for MobileNetV2.
    Output range is approximately [-1, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, target_size)

    img_float = img_resized.astype(np.float32)

    # MobileNetV2 preprocessing
    img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_float
    )

    img_batch = np.expand_dims(img_processed, axis=0)

    return img_batch


def get_data_augmentation_layers():

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ], name="data_augmentation")


def create_dataset_pipelines(
    dataset_dir="dataset",
    target_size=DEFAULT_TARGET_SIZE,
    batch_size=16
):

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    print(f"Loading training dataset from: {train_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_size,
        shuffle=True,
        seed=42
    )

    print("Class names:", train_ds.class_names)

    print(f"Loading validation dataset from: {val_dir}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False
    )

    print("Validation class names:", val_ds.class_names)

    def preprocess(images, labels):
        images = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.cast(images, tf.float32)
        )
        return images, labels

    train_ds = train_ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds