import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduce TensorFlow overhead

import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("mask_detector_mobilenet.h5", compile=False, safe_mode=False)
print("Model loaded!")

# Camera start (stable)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

print("Camera started!")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize + brightness
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=50)

    # Gray + contrast improve
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Keep only largest face
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        faces = [faces[0]]

    for (x, y, w, h) in faces:
        if w < 100 or h < 100:
            continue

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        # Preprocess
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        # Predict
        prediction = model.predict(face, verbose=0)[0][0]

        if prediction < 0.5:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Mask Detection FINAL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()