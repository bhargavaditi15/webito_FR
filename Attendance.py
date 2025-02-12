import cv2
import numpy as np
import mysql.connector
import pickle
import csv
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Connect to MySQL
conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
cursor = conn.cursor()

# Load face detection model
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load stored face data from MySQL
cursor.execute("SELECT name, face_data FROM faces")
records = cursor.fetchall()

if not records:
    print("No face data found in database!")
    exit()

# Extract names and face data
LABELS = []
FACES = []

for name, face_blob in records:
    face_array = pickle.loads(face_blob)
    LABELS.extend([name] * len(face_array))
    FACES.extend(face_array)

FACES = np.array(FACES).reshape(len(LABELS), -1)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Get current date and time
date = datetime.now().strftime("%Y-%m-%d")
day = datetime.now().strftime("%A")
timestamp = datetime.now().strftime("%H:%M:%S")

# Start video capture
video = cv2.VideoCapture(0)
ret, frame = video.read()

if not ret:
    print("Error: Could not capture image.")
    video.release()
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

if len(faces) == 0:
    print("No face detected.")
    person_name = "Unknown"
    status = "-"
else:
    for (x, y, w, h) in faces:
        live_face = frame[y:y+h, x:x+w]
        live_face_resized = cv2.resize(live_face, (50, 50)).flatten().reshape(1, -1)

        # Compare similarity
        similarity = cosine_similarity(live_face_resized, live_face_resized)[0][0]
        print(f"Face Similarity Score: {similarity:.2f}")

        threshold = 0.75
        if similarity >= threshold:
            try:
                person_name = knn.predict(live_face_resized)[0]
                status = "Present"
            except Exception:
                person_name = "Unknown"
                status = "-"
        else:
            person_name = "Unknown"
            status = "-"

# Insert attendance into MySQL
cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (%s, %s, %s, %s, %s)",
               (date, day, str(person_name), timestamp, status))

conn.commit()

print(f"Attendance Marked: {person_name} - {status} at {timestamp} on {date} ({day})")

# Close database connection
cursor.close()
conn.close()

# Release video capture
video.release()
cv2.destroyAllWindows()
