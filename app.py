from flask import Flask, render_template, request, jsonify
import mysql.connector
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__)

# Database connection
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="15052003",
            database="FaceRecognition"
        )
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Ensure tables exist
conn = get_db_connection()
if conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            image_path VARCHAR(255) NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATE NOT NULL,
            day VARCHAR(20) NOT NULL,
            name VARCHAR(255) NOT NULL,
            time TIME NOT NULL,
            status VARCHAR(50) NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Load known faces from the database
def load_known_faces():
    conn = get_db_connection()
    if not conn:
        return [], []

    cursor = conn.cursor()
    cursor.execute("SELECT name, image_path FROM face")
    known_faces = []
    known_names = []

    for name, image_path in cursor.fetchall():
        if os.path.exists(image_path):
            img = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
        else:
            print(f"Warning: Missing file {image_path}")

    conn.close()
    return known_faces, known_names

# Recognize face from frame
def recognize_face(frame):
    known_faces, known_names = load_known_faces()
    if not known_faces:
        return "Unknown"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            return known_names[best_match_index]

    return "Unknown"

@app.route('/')
def home():
    conn = get_db_connection()
    if not conn:
        return "Database connection failed!", 500

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC LIMIT 5")
    records = cursor.fetchall()
    conn.close()
    return render_template('index.html', records=records)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        image_path = f"faces/{name}.jpg"

        # Ensure 'faces' directory exists
        os.makedirs("faces", exist_ok=True)

        # Capture image from webcam
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()

        if not ret:
            return jsonify({"error": "Failed to capture image!"}), 400

        cv2.imwrite(image_path, frame)  # Save image

        # Store in database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed!"}), 500

        cursor = conn.cursor()
        cursor.execute("INSERT INTO face (name, image_path) VALUES (%s, %s)", (name, image_path))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Face data for {name} stored successfully!"})

    return render_template('register.html')

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        return jsonify({"error": "Error capturing image!"}), 400

    name = recognize_face(frame)  # Recognize face

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed!"}), 500

    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (NOW(), DAYNAME(NOW()), %s, NOW(), 'Present')", (name,))
    conn.commit()
    conn.close()

    return jsonify({"message": f"Attendance marked for {name}!"})

@app.route('/attendance-records')
def attendance_records():
    conn = get_db_connection()
    if not conn:
        return "Database connection failed!", 500

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template('records.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
