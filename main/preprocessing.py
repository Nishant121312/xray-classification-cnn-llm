import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
from pathlib import Path
from io import BytesIO

model_path = Path('models/vgg16_tl.keras')
image_model = tf.keras.models.load_model(model_path)
class_names = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']


def get_predicted_label(prediction):
    predicted_label = class_names[np.argmax(prediction)]
    print("Predicted Label:", predicted_label)
    return predicted_label


def predict_image_label(uploaded_image):
    # Preprocess the image
    image_data = uploaded_image.read()
    pil_image = Image.open(BytesIO(image_data))
    img = np.array(pil_image)

    if img.ndim == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    img = cv.resize(img, (180, 180))
    img = img / .255
    img = np.expand_dims(img, axis=0)

    # Make prediction using the image model
    prediction = image_model.predict([img])
    print(prediction)

    # Get the predicted label
    predicted_label = get_predicted_label(prediction)

    return predicted_label
