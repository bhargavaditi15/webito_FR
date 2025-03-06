# import cv2
# import numpy as np
# import pickle
# import mysql.connector
# from flask import Flask, render_template, Response
# from sklearn.preprocessing import LabelEncoder  # Add this import

# app = Flask(__name__)

# # Database Connection
# conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
# cursor = conn.cursor()

# # Load face recognition data
# cursor.execute("SELECT name, face_data FROM faces")
# records = cursor.fetchall()

# if not records:
#     print("No face data found in database!")
#     exit()

# # Prepare the model
# LABELS, FACES = [], []
# for name, face_blob in records:
#     face_array = pickle.loads(face_blob)
#     LABELS.extend([name] * len(face_array))
#     FACES.extend(face_array)

# FACES = np.array(FACES).reshape(len(LABELS), -1)

# # Encode string labels into numbers
# label_encoder = LabelEncoder()
# LABELS_NUMERIC = label_encoder.fit_transform(LABELS)  # Convert names to numbers

# # Initialize face detector & classifier
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# knn = cv2.ml.KNearest_create()

# # Train the KNN model using numeric labels
# knn.train(FACES.astype(np.float32), cv2.ml.ROW_SAMPLE, LABELS_NUMERIC.astype(np.int32))

# # Function to recognize faces in the video feed
# def recognize_face(face_resized):
#     _, results, _, _ = knn.findNearest(face_resized.astype(np.float32), k=5)
#     predicted_label = int(results[0][0])
#     person_name = label_encoder.inverse_transform([predicted_label])[0]  # Convert back to name
#     return person_name

# # CCTV RTSP URL or USB Camera
# CCTV_URL = "rtsp://username:password@your_cctv_ip:port"  # Replace with your CCTV RTSP URL
# cap = cv2.VideoCapture(CCTV_URL)  # Use 0 for a USB webcam

# def generate_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             face_crop = frame[y:y+h, x:x+w]
#             face_resized = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)

#             # Recognize face
#             person_name = recognize_face(face_resized)

#             # Draw rectangle & label
#             color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#             cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         # Encode frame for streaming
#         _, buffer = cv2.imencode(".jpg", frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template("index.htm")

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)


# import cv2
# import numpy as np
# import pickle
# import mysql.connector
# from flask import Flask, render_template, Response, request, redirect

# app = Flask(__name__)

# # Database Connection
# conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
# cursor = conn.cursor()

# # Initialize face detector
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Initialize Video Capture
# CCTV_URL = "rtsp://username:password@your_cctv_ip:port"  # Replace with your CCTV RTSP URL
# cap = cv2.VideoCapture(CCTV_URL)  # Use 0 for a USB webcam

# def generate_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         _, buffer = cv2.imencode(".jpg", frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture', methods=['POST'])
# def capture():
#     name = request.form['name']
#     face_data = []

#     video = cv2.VideoCapture(0)  # Open webcam for face capture

#     while len(face_data) < 100:
#         ret, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w]
#             resized_img = cv2.resize(crop_img, (50, 50))
#             face_data.append(resized_img)

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             cv2.putText(frame, f"Capturing: {len(face_data)}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             cv2.imshow("Capturing Face Data", frame)
#             cv2.waitKey(1)

#     video.release()
#     cv2.destroyAllWindows()

#     # Store face data in database
#     face_data_np = np.array(face_data)
#     face_data_bytes = pickle.dumps(face_data_np)
#     cursor.execute("INSERT INTO faces (name, face_data) VALUES (%s, %s)", (name, face_data_bytes))
#     conn.commit()

#     return redirect('/')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)

# from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
# import cv2
# import numpy as np
# import mysql.connector
# import pickle
# import os
# from datetime import datetime
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# import csv
# import time

# app = Flask(__name__)

# # Database connection
# conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
# cursor = conn.cursor()

# # Initialize OpenCV face detector
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Open video capture
# video = cv2.VideoCapture(0)  # Change index if using external camera (try 1, 2, etc.)

# # Global Variables
# collecting_data = False
# face_data = []
# current_name = ""

# # Function to capture video frames
# def generate_frames():
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         else:
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Route to serve video stream
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route for homepage
# @app.route('/')
# def index():
#     return render_template('index.htm')

# # Route to start capturing face data
# @app.route('/start_capture', methods=['POST'])
# def start_capture():
#     global collecting_data, face_data, current_name

#     current_name = request.form['name']
#     face_data = []
#     collecting_data = True

#     return jsonify({"message": "Face data collection started"})

# # Route to capture face data
# @app.route('/capture')
# def capture():
#     global collecting_data, face_data, current_name

#     if not collecting_data:
#         return jsonify({"message": "Data collection not started yet"})

#     ret, frame = video.read()
#     if not ret:
#         return jsonify({"error": "Could not capture image"})

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         resized_img = cv2.resize(crop_img, (50, 50))
#         face_data.append(resized_img)

#         if len(face_data) >= 100:
#             collecting_data = False
#             save_face_data(current_name, face_data)
#             return jsonify({"message": f"Face data for {current_name} stored successfully!"})

#     return jsonify({"message": f"Captured {len(face_data)}/100 images"})

# # Function to save face data to MySQL
# def save_face_data(name, face_data):
#     face_data_np = np.array(face_data)
#     face_data_bytes = pickle.dumps(face_data_np)

#     cursor.execute("INSERT INTO faces (name, face_data) VALUES (%s, %s)", (name, face_data_bytes))
#     conn.commit()

# # Route to mark attendance
# @app.route('/mark_attendance')
# def mark_attendance():
#     global video

#     # Load stored face data
#     cursor.execute("SELECT name, face_data FROM faces")
#     records = cursor.fetchall()

#     if not records:
#         return jsonify({"error": "No face data found in database!"})

#     LABELS = []
#     FACES = []

#     for name, face_blob in records:
#         face_array = pickle.loads(face_blob)
#         LABELS.extend([name] * len(face_array))
#         FACES.extend(face_array)

#     FACES = np.array(FACES).reshape(len(LABELS), -1)

#     # Train KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(FACES, LABELS)

#     ret, frame = video.read()
#     if not ret:
#         return jsonify({"error": "Could not capture image"})

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     if len(faces) == 0:
#         person_name = "Unknown"
#         status = "-"
#     else:
#         for (x, y, w, h) in faces:
#             live_face = frame[y:y+h, x:x+w]
#             live_face_resized = cv2.resize(live_face, (50, 50)).flatten().reshape(1, -1)
#             similarity = cosine_similarity(live_face_resized, live_face_resized)[0][0]

#             threshold = 0.75
#             if similarity >= threshold:
#                 person_name = knn.predict(live_face_resized)[0]
#                 status = "Present"
#             else:
#                 person_name = "Unknown"
#                 status = "-"

#     # Insert attendance record
#     date = datetime.now().strftime("%Y-%m-%d")
#     time_now = datetime.now().strftime("%H:%M:%S")
#     day = datetime.now().strftime("%A")

#     cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (%s, %s, %s, %s, %s)",
#                    (date, day, str(person_name), time_now, status))
#     conn.commit()

#     return jsonify({"message": f"Attendance Marked: {person_name} - {status}"})

# # Run Flask App
# if __name__ == '__main__':
#     app.run(debug=True)


import cv2
import numpy as np
import pickle
import mysql.connector
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv

app = Flask(__name__)

# Database Connection
conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
cursor = conn.cursor()

# Load Face Recognition Data
cursor.execute("SELECT name, face_data FROM faces")
records = cursor.fetchall()

LABELS, FACES = [], []
for name, face_blob in records:
    face_array = pickle.loads(face_blob)
    LABELS.extend([name] * len(face_array))
    FACES.extend(face_array)

FACES = np.array(FACES).reshape(len(LABELS), -1)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Face Detector
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Video Capture
CCTV_URL = "rtsp://username:password@your_cctv_ip:port"
cap = cv2.VideoCapture(0)  # Change to CCTV_URL for CCTV feed

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)
            
            try:
                person_name = knn.predict(face_resized)[0]
            except:
                person_name = "Unknown"
            
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    date = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not capture image."})
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return jsonify({"status": "No face detected"})
    
    for (x, y, w, h) in faces:
        live_face = frame[y:y+h, x:x+w]
        live_face_resized = cv2.resize(live_face, (50, 50)).flatten().reshape(1, -1)
        
        similarity = cosine_similarity(live_face_resized, live_face_resized)[0][0]
        threshold = 0.75
        if similarity >= threshold:
            try:
                person_name = knn.predict(live_face_resized)[0]
                status = "Present"
            except:
                person_name = "Unknown"
                status = "-"
        else:
            person_name = "Unknown"
            status = "-"
    
    cursor.execute("INSERT INTO attendance (date, day, name, time, status) VALUES (%s, %s, %s, %s, %s)",
                   (date, day, str(person_name), timestamp, status))
    conn.commit()
    
    return jsonify({"name": person_name, "status": status, "time": timestamp})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
