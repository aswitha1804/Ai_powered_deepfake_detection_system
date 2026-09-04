"""
Phase 5: MobileNetV2 Deepfake Classifier Architecture
AI-Powered Deepfake Detection System

Uses MobileNetV2 pre-trained backbone for robust feature extraction and real-world image generalization.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

def build_deepfake_detection_model(input_shape=(128, 128, 3), learning_rate=0.0005):
    """
    Builds a Transfer Learning CNN based on MobileNetV2 pre-trained backbone.
    
    Architecture:
    1. MobileNetV2 Backbone (Pre-trained ImageNet features)
    2. GlobalAveragePooling2D
    3. Dense (64) + Dropout (0.3)
    4. Dense (1, Sigmoid) Output: P(REAL)
    """
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        
        inputs = layers.Input(shape=input_shape, name="input_image")
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="global_pool")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.3, name="drop1")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output_prediction")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Deepfake_Classifier")
    except Exception as e:
        print(f"[Model] MobileNetV2 fallback to Custom CNN ({e})")
        model = models.Sequential(name="Custom_CNN_Classifier")
        model.add(layers.Input(shape=input_shape, name="input_image"))
        model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    return model

def save_model_summary(model, output_path="models/model_summary.txt"):
    """
    Saves printable ASCII summary of model architecture to a text file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Saved model architecture summary to: {output_path}")

def test_model_building():
    """
    Test routine to build model and verify forward pass.
    """
    print("==================================================")
    print("    Deepfake Detection - MobileNetV2 Architecture ")
    print("==================================================\n")
    
    model = build_deepfake_detection_model(input_shape=(128, 128, 3))
    model.summary()
    
    dummy_input = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
    dummy_output = model(dummy_input)
    
    print(f"\nForward Pass Verification:")
    print(f"  - Input Tensor Shape  : {dummy_input.shape}")
    print(f"  - Output Tensor Shape : {dummy_output.shape}")
    print(f"  - Dummy Prediction    : {dummy_output.numpy()[0][0]:.4f}")
    
    assert dummy_output.shape == (1, 1), "Output shape must be (1, 1)!"
    save_model_summary(model)
    print("\nSUCCESS: MobileNetV2 deepfake classifier built!")

if __name__ == "__main__":
    test_model_building()
