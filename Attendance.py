import cv2
import numpy as np
import mysql.connector
import pickle
import csv
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

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

    # Convert to grayscale for better accuracy
    face_array_gray = np.array([cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in face_array])

    LABELS.extend([name] * len(face_array_gray))
    FACES.extend(face_array_gray)

FACES = np.array(FACES).reshape(len(LABELS), -1)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Reduced neighbors for better accuracy
knn.fit(FACES, LABELS)

# Get current date and time
date = datetime.now().strftime("%Y-%m-%d")
day = datetime.now().strftime("%A")
timestamp = datetime.now().strftime("%H:%M:%S")

# Start video capture
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open camera")
    exit()

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
        live_face_gray = cv2.cvtColor(live_face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        live_face_resized = cv2.resize(live_face_gray, (50, 50)).flatten().reshape(1, -1)

        try:
            person_name = knn.predict(live_face_resized)[0]
            status = "Present"
        except Exception:
            person_name = "Unknown"
            status = "-"

# Insert attendance into MySQL
cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (%s, %s, %s, %s, %s)",
               (date, day, str(person_name), timestamp, status))

conn.commit()

print(f"Attendance Marked: {person_name} - {status} at {timestamp} on {date} ({day})")

# Fetch all attendance records from the 'attendance' table
cursor.execute("SELECT * FROM attendance")
attendance_records = cursor.fetchall()

# Check if any records are fetched
if not attendance_records:
    print("No attendance records found!")
else:
    # Open a CSV file to write the attendance records
    with open("attendance_data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Date", "Day", "Name", "Time", "Status"])  # Header row
        writer.writerows(attendance_records)  # Write all records

    print("Attendance data has been written to attendance_data.csv")

# Close database connection
cursor.close()
conn.close()

# Release video capture
video.release()
cv2.destroyAllWindows()


