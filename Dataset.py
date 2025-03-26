import cv2
import numpy as np
import os
import mysql.connector
import pickle

# Initialize database connection
conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
cursor = conn.cursor()

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your name: ")

face_data = []

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        face_data.append(resized_img)

        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    if len(face_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to binary
face_data_np = np.array(face_data)
face_data_bytes = pickle.dumps(face_data_np)

# Insert into database
cursor.execute("INSERT INTO faces (name, face_data) VALUES (%s, %s)", (name, face_data_bytes))
conn.commit()

print(f"Face data for {name} stored in the database.")

cursor.close()
conn.close()