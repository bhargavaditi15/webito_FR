from flask import Flask, request, jsonify
import cv2
import numpy as np
import mysql.connector
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Database connection
conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
cursor = conn.cursor()

# Load face detection model
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# **1. Face Registration API**
@app.route('/register-face', methods=['POST'])
def register_face():
    name = request.json.get('name')
    if not name:
        return jsonify({"error": "Name is required"}), 400

    video = cv2.VideoCapture(0)
    face_data = []

    while len(face_data) < 10:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cropped_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (50, 50))
            face_data.append(resized_img)

    video.release()
    cv2.destroyAllWindows()

    face_data_np = np.array(face_data)
    face_data_bytes = pickle.dumps(face_data_np)

    # Insert into database
    cursor.execute("INSERT INTO faces (name, face_data) VALUES (%s, %s)", (name, face_data_bytes))
    conn.commit()

    return jsonify({"message": f"Face data for {name} stored successfully"}), 200


# **2. Attendance Marking API**
@app.route('/mark-attendance', methods=['GET'])
def mark_attendance():
    # Load stored face data
    cursor.execute("SELECT name, face_data FROM faces")
    records = cursor.fetchall()

    if not records:
        return jsonify({"error": "No face data found in database"}), 400

    LABELS, FACES = [], []
    for name, face_blob in records:
        face_array = pickle.loads(face_blob)
        LABELS.extend([name] * len(face_array))
        FACES.extend(face_array)

    FACES = np.array(FACES).reshape(len(LABELS), -1)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    video = cv2.VideoCapture(0)
    ret, frame = video.read()

    if not ret:
        video.release()
        return jsonify({"error": "Could not capture image"}), 500

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        video.release()
        return jsonify({"message": "No face detected"}), 400

    for (x, y, w, h) in faces:
        live_face = frame[y:y+h, x:x+w]
        live_face_resized = cv2.resize(live_face, (50, 50)).flatten().reshape(1, -1)

        similarity = cosine_similarity(live_face_resized, live_face_resized)[0][0]
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

    video.release()
    cv2.destroyAllWindows()

    # Insert attendance into MySQL
    date, day, timestamp = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%A"), datetime.now().strftime("%H:%M:%S")
    cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (%s, %s, %s, %s, %s)",
                   (date, day, person_name, timestamp, status))
    conn.commit()

    return jsonify({"message": f"Attendance marked: {person_name} - {status} at {timestamp} on {date} ({day})"}), 200


# **Run Flask Server**
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
