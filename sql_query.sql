CREATE DATABASE FaceRecognition;

USE FaceRecognition;

CREATE TABLE faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    face_data LONGBLOB NOT NULL
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    day VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    time TIME NOT NULL,
    status VARCHAR(20) NOT NULL
);

SELECT * FROM attendance;
SELECT * FROM faces