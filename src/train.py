"""
Phase 6: Model Training Pipeline
AI-Powered Deepfake Detection System
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_deepfake_detection_model
from preprocess import create_dataset_pipelines

# ─── SETTINGS ─────────────────────────────────────────────────────────────────
# Paths relative to the PROJECT ROOT (deepfake-detection-system/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

IMG_SIZE   = (128, 128)
BATCH_SIZE = 16
EPOCHS     = 25

MODEL_PATH = os.path.join(MODEL_DIR, "best_deepfake_model.keras")
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── LOAD DATASET ─────────────────────────────────────────────────────────────
print("\nLoading dataset...")
train_ds, val_ds = create_dataset_pipelines(
    dataset_dir=DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ─── BUILD MODEL ──────────────────────────────────────────────────────────────
print("\nBuilding MobileNetV2 model...")
model = build_deepfake_detection_model(input_shape=(128, 128, 3), learning_rate=0.0001)
model.summary()


# ─── CALLBACKS ────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(MODEL_DIR, "training_log.csv"),
        append=False
    )
]


# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nStarting training...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ─── EVALUATE ─────────────────────────────────────────────────────────────────
print("\nEvaluating model on validation set...")
results = model.evaluate(val_ds, verbose=1)
print("\nValidation Results:")
for name, value in zip(model.metrics_names, results):
    print(f"  {name}: {value:.4f}")


# ─── SAVE TRAINING CURVES ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(MODEL_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nTraining curves saved -> {curves_path}")

print(f"\nTraining complete! Best model saved at:\n   {MODEL_PATH}")