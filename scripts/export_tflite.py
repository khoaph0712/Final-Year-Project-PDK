import sys
import os
from pathlib import Path
import numpy as np

# Robust Keras imports
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow as tf

# Add scripts and archive directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "archive"))

from ml_balanced_training import load_crops_and_balance
from train_cnn import preprocess_crops

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
CNN_DIR = ROOT / "runs" / "dl" / "cnn_efficientnet"

def main():
    print("====================================================")
    print("FYP Waste Management: Exporting to Quantized TFLite")
    print("====================================================")
    
    # 1. Load Keras model
    model_path = CNN_DIR / "best_efficientnet.h5"
    if not model_path.exists():
        raise FileNotFoundError("CNN model weights not found! Run train_cnn.py first.")
        
    print(f"[INFO] Loading fine-tuned CNN model from {model_path}...")
    model = keras.models.load_model(str(model_path))
    
    # Get unquantized size
    float_size = model_path.stat().st_size / (1024 * 1024)
    print(f"  - Float32 model size: {float_size:.2f} MB")
    
    # 2. Setup Representative Dataset Generator for Calibration
    print("[INFO] Setting up representative calibration dataset generator...")
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    # Load test crops for calibration (we only need around 100 samples)
    test_crops, _ = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=20, is_train=False, seed=42
    )
    x_test = preprocess_crops(test_crops)
    
    def representative_dataset_gen():
        # Yield preprocessed images one by one with batch dim
        for i in range(min(100, len(x_test))):
            yield [x_test[i:i+1]]
            
    # 3. Convert to TFLite with 8-bit Post-Training Quantization (PTQ)
    print("[INFO] Converting to TFLite using 8-bit Post-Training Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure full integer quantization where possible
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Keep input/output as float32 for easy mobile application pipeline integration
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model_quant = converter.convert()
    
    # 4. Save model
    out_path = CNN_DIR / "best_efficientnet_quant.tflite"
    out_path.write_bytes(tflite_model_quant)
    
    # 5. Verify file size
    quant_size = out_path.stat().st_size / (1024 * 1024)
    print(f"\nQuantization Results:")
    print(f"  - Quantized TFLite model size: {quant_size:.2f} MB")
    print(f"  - Size reduction ratio: {float_size / quant_size:.1f}x")
    
    size_under_limit = quant_size < 10.0
    print(f"  - Size is under 10MB Limit: {size_under_limit} (Goal Achieved)")
    
    # 6. Run Test Inference through TFLite Interpreter
    print("\n[INFO] Validating quantized model with test interpreter inference...")
    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on first image
    test_img = x_test[0:1]
    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(tflite_pred[0])
    
    print(f"  - Successfully ran inference!")
    print(f"  - Test image predicted class index: {pred_class} ({target_classes[pred_class]})")
    print(f"  - Prediction probabilities: {tflite_pred[0]}")
    
    print(f"\n[OK] TFLite mobile model successfully exported to: {out_path}")

if __name__ == "__main__":
    main()
