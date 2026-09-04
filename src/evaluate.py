"""
Phase 7: Model Evaluation & Performance Metrics Module
AI-Powered Deepfake Detection System

This module handles:
1. Loading the trained Keras model (models/best_deepfake_model.keras).
2. Running evaluation predictions on the validation set.
3. Calculating Accuracy, Precision, Recall, F1-Score, and Confusion Matrix using Scikit-Learn.
4. Generating and saving a Confusion Matrix visualization plot (models/confusion_matrix.png).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import create_dataset_pipelines

DEFAULT_MODEL_PATH = os.path.join("models", "best_deepfake_model.keras")
CM_PLOT_SAVE_PATH = os.path.join("models", "confusion_matrix.png")

def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads saved Keras model file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found at: {model_path}. Please run src/train.py first.")
        
    print(f"[Evaluation] Loading trained CNN model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def plot_confusion_matrix(cm, class_names=["Fake (0)", "Real (1)"], output_path=CM_PLOT_SAVE_PATH):
    """
    Plots and saves a styled Confusion Matrix diagram.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Deepfake Detection Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)

    # Print numerical cell counts inside the grid boxes
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold"
            )

    plt.ylabel("Actual True Label")
    plt.xlabel("Model Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix plot to: {output_path}")

def evaluate_model_performance(model_path=DEFAULT_MODEL_PATH, dataset_dir="dataset", threshold=0.5):
    """
    Evaluates trained model on validation dataset and prints detailed metrics.
    """
    print("==================================================")
    print("    Deepfake Detection - Model Evaluation Suite   ")
    print("==================================================\n")
    
    # Step 1: Load model
    model = load_trained_model(model_path)
    
    # Step 2: Load validation dataset pipeline
    _, val_ds = create_dataset_pipelines(dataset_dir=dataset_dir, batch_size=16)
    
    # Step 3: Collect ground truth labels and predictions
    y_true = []
    y_pred_probs = []
    
    print("\nRunning inference predictions on validation set...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_probs.extend(preds.flatten())
        
    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    
    # Apply classification probability threshold (default 0.5)
    # raw_prob >= 0.5 is class 1 (REAL)
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    
    # Step 4: Compute Metrics
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels, zero_division=0)
    rec = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_labels)
    
    # Step 5: Print Evaluation Summary
    print("\n" + "=" * 50)
    print("           MODEL PERFORMANCE METRICS SUMMARY       ")
    print("=" * 50)
    print(f"  - Accuracy        : {acc * 100:.2f}%")
    print(f"  - Precision       : {prec * 100:.2f}%")
    print(f"  - Recall          : {rec * 100:.2f}%")
    print(f"  - F1-Score        : {f1 * 100:.2f}%")
    print("-" * 50)
    print("\nDetailed Scikit-Learn Classification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=["Fake (0)", "Real (1)"], zero_division=0))
    
    print("Confusion Matrix:")
    print(f"  True Fake  (TN): {cm[0,0]:<4} | False Real (FP): {cm[0,1]:<4}")
    print(f"  False Fake (FN): {cm[1,0]:<4} | True Real  (TP): {cm[1,1]:<4}")
    print("=" * 50)
    
    # Step 6: Save Confusion Matrix Diagram
    plot_confusion_matrix(cm)
    
    print("\nSUCCESS: Model evaluation complete!")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

if __name__ == "__main__":
    evaluate_model_performance()

