import tensorflow as tf
import gdown
from PIL import Image
import numpy as np
import cv2 as cv
from pathlib import Path
from io import BytesIO

# Google Drive file ID and download URL
file_id = "1m7f6VILJx4du4cxj5vEA-u3Gm6A5QNBO"
url = f"https://drive.google.com/uc?id={file_id}"

# Define model path
model_path = Path("models/vgg16_tl.keras")

# Create the models directory if it doesn't exist
model_path.parent.mkdir(parents=True, exist_ok=True)

# Download the model if it does not exist
if not model_path.exists():
    print("Downloading model from Google Drive...")
    gdown.download(url, str(model_path), quiet=False)

# Load the model
image_model = tf.keras.models.load_model(model_path)

# Class names
class_names = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

def get_predicted_label(prediction):
    predicted_label = class_names[np.argmax(prediction)]
    print("Predicted Label:", predicted_label)
    return predicted_label

def predict_image_label(uploaded_image):
    # Read and preprocess the image
    image_data = uploaded_image.read()
    pil_image = Image.open(BytesIO(image_data))
    img = np.array(pil_image)

    # Convert grayscale to RGB if necessary
    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    # Resize and normalize image
    img = cv.resize(img, (180, 180))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = image_model.predict(img)
    print("Raw prediction:", prediction)

    # Get the predicted label
    predicted_label = get_predicted_label(prediction)

    return predicted_label
