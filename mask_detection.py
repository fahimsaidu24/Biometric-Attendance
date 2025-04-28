
# Your full Python code goes here
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            resized = cv2.resize(face, (128, 128)) / 255.0
            reshaped = np.reshape(resized, (1, 128, 128, 3))
            prediction = model.predict(reshaped)[0][0]

            if prediction < 0.5:
                label = "Mask"
                color = (0, 0, 255)
                prompt = "Please remove mask to scan"
            else:
                label = "No Mask"
                color = (0, 255, 0)
                prompt = "Proceeding to face recognition..."

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass

    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
