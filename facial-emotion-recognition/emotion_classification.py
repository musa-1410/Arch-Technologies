import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')

# Emotion labels
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess for emotion detection
        cropped_img = cv2.resize(face_roi, (48, 48))
        cropped_img = cropped_img.reshape(1, 48, 48, 1) / 255.0
        
        # Predict emotion
        emotion_pred = emotion_model.predict(cropped_img)
        emotion_label = EMOTIONS[np.argmax(emotion_pred)]
        confidence = np.max(emotion_pred)
        
        # Display results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion_label} ({confidence:.2f})"
        cv2.putText(
            frame, text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    
    cv2.imshow('Facial Emotion Recognition', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()