import tensorflow as tf
from pathlib import Path

model_path = Path("C:/zehr/my-streamlit-app/models/vgg16_tl.keras")

if not model_path.exists():
    print("❌ Model file not found!")
else:
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Model loading failed:", str(e))
