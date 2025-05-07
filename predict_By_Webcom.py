import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load your trained model
model = load_model("emotion_model_finetuned.keras")

# Emotion class names (make sure the order matches your model)
class_names = ['happy', 'sad']

# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optional: set higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
predict_every_n_frames = 10
last_prediction = None

# Minimum face box size to avoid false detections (tweak if needed)
MIN_FACE_SIZE = 80

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Check face box size (quality filter)
                if w_box < MIN_FACE_SIZE or h_box < MIN_FACE_SIZE:
                    continue  # Skip tiny or bad detections

                # Only predict every N frames
                if frame_count % predict_every_n_frames == 0:
                    try:
                        face_roi = frame[y:y+h_box, x:x+w_box]
                        face_resized = cv2.resize(face_roi, (260, 260))
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        face_preprocessed = preprocess_input(face_rgb)
                        input_tensor = np.expand_dims(face_preprocessed, axis=0)

                        preds = model.predict(input_tensor, verbose=0)
                        class_index = np.argmax(preds)
                        confidence = np.max(preds)
                        last_prediction = (class_names[class_index], confidence)

                    except Exception as e:
                        print("Error in prediction:", e)
                        continue

                # Draw bounding box and label
                if last_prediction:
                    emotion, conf = last_prediction
                    label = f"{emotion} ({conf*100:.1f}%)"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        frame_count += 1
        cv2.imshow("Real-Time Emotion Classifier", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
