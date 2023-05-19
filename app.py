import cv2
import face_recognition
import numpy as np
from deepface import DeepFace

# Load pre-trained models for face detection and emotion recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to process each frame of the video stream
def process_frame(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_img = frame[y:y+h, x:x+w]

        # Perform emotion recognition on the face ROI
        emotions = DeepFace.analyze(face_img, actions=['emotion'])

        # Extract age and gender information from the face ROI
        age_gender_info = DeepFace.analyze(face_img, actions=['age', 'gender'])

        # Get the predicted emotion, age, and gender
        emotion = max(emotions['emotion'], key=emotions['emotion'].get)
        age = int(age_gender_info['age'])
        gender = age_gender_info['gender']

        # Calculate the percentage of certainty for emotion, age, and gender
        emotion_percent = emotions['emotion'][emotion] * 100
        age_percent = 100 - abs(age_gender_info['age_range'] - age) * 10
        gender_percent = age_gender_info['gender_probability'][gender] * 100

        # Display the information on the frame
        cv2.putText(frame, f"Emotion: {emotion} ({emotion_percent:.2f}%)", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age} ({age_percent:.2f}%)", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {gender} ({gender_percent:.2f}%)", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Live Face Analysis', frame)

# Open the video capture
cap = cv2.VideoCapture(0)

# Continuously process frames from the video stream
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Process the frame
    process_frame(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
