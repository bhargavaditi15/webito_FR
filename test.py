import cv2

cap = cv2.VideoCapture(0)  # Change 0 to 1 or 2 if using an external webcam

if not cap.isOpened():
    print("Error: Camera not detected")
else:
    print("Camera is working!")

cap.release()
