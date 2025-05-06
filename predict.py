import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
model = load_model("emotion_model.h5")

# Class names must match training order
class_names = ['happy', 'sad']  # adjust if needed
img = r'C:\Users\HP\Downloads\EmotionsClassifier\test_img\saaad.jpg'
def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (260, 260))  # Resize to match model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)  # Important if you used EfficientNet preprocessing
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(img)
    class_index = np.argmax(preds, axis=1)[0]
    print(f"Prediction: {class_names[class_index]} (Confidence: {np.max(preds)*100:.2f}%)")

# Example usage
predict_image(img)  # Replace with your image path
