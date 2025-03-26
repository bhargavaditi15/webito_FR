# import secrets
# print(secrets.token_hex(16))  # Generates a 32-character hex key

# # 18b8e3405221ae9e3ba93665ff301f18

# import dlib
# import face_recognition

# print("dlib and face_recognition installed successfully!")

import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="15052003",
        database="FaceRecognition"
    )
    print("Connected successfully!")
    conn.close()
except mysql.connector.Error as err:
    print(f"Error: {err}")
