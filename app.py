from flask import Flask, render_template, request, jsonify
import subprocess
import mysql.connector

app = Flask(__name__)

# Database connection
conn = mysql.connector.connect(host="localhost", user="root", password="15052003", database="FaceRecognition")
cursor = conn.cursor()

# Ensure the face table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS face (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        image_path VARCHAR(255) NOT NULL
    )
""")
conn.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        
        # Insert into database (dummy path for now)
        image_path = f"faces/{name}.jpg"
        cursor.execute("INSERT INTO face (name, image_path) VALUES (%s, %s)", (name, image_path))
        conn.commit()
        
        # Capture face data using Dataset.py
        subprocess.run(['python', 'Dataset.py'], input=name.encode())
        
        return jsonify({"message": "Face data captured and stored in database!"})
    
    return render_template('register.html')

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    subprocess.run(['python', 'Attendance.py'])
    return jsonify({"message": "Attendance marked!"})

@app.route('/attendance-records')
def attendance_records():
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    return render_template('records.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
