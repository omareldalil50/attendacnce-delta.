# Import necessary libraries
import cv2
import pandas as pd
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import numpy as np
import base64
from datetime import datetime
import os

# Load the Viola-Jones face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load images and student names
# تحميل الصور وأسماء الطلاب
images = [
    {"name": "Omar El-Dalil", "image_path": "images/omar.jpg"},
    {"name": "Diab", "image_path": "images/Diab.jfif"},
    {"name": "Mostafa Hagag", "image_path": "images/Mostafa Hagag.jfif"},
    {"name": "Ahmed Sheaba", "image_path": "images/Ahmed Sheaba.jfif"},
    {"name": "Dr.Amany Sarhan", "image_path": "images/Dr.Amany.jpg"},
    {"name": "Basel Darwish", "image_path": "images/Basel.jpg"},
    {"name": "Hussein El-Sabagh", "image_path": "images/Hussien.jpg"},
]
# File name for the attendance log
attendance_file = "attendance_log.xlsx"

# Create an Excel file if not exists
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_excel(attendance_file, index=False, engine='openpyxl')

attendance_df = pd.DataFrame(columns=["Name", "Time", "Present"])

known_faces = []
known_names = []

for student in images:
    image = face_recognition.load_image_file(student["image_path"])
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(student["name"])

# Initialize Flask app
app = Flask(__name__)

# Create a route for the web application
@app.route('/')
def index():
    return render_template('index.html')

# Add a new route for downloading the attendance log
@app.route('/download_attendance', methods=['GET'])
def download_attendance():
    # Generate the Excel file
    generate_excel_file()

    # Send the file as a downloadable response
    return send_file(attendance_file, as_attachment=True)

def generate_excel_file():
    # Read the existing attendance log
    try:
        attendance_df = pd.read_excel(attendance_file, engine='openpyxl')
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time"])

    # Save the attendance log to the Excel file
    attendance_df.to_excel(attendance_file, index=False, engine='openpyxl')

# Create a route for uploading an image
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded file to a temporary location
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # Call your existing face recognition function
    result_image, accuracy, recognized_names = recognize_faces_in_image(temp_path)

    # Log the attendance
    log_attendance(recognized_names)

    # Convert result image to base64-encoded string
    result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()

    # Display the result and accuracy
    return render_template('result.html', result_image=result_image_base64, accuracy=accuracy, recognized_names=recognized_names)

# Create a route for capturing an image
@app.route('/capture', methods=['POST'])
def capture():
    # Open a connection to the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Capture a single frame
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Save the captured frame to a temporary location
    temp_path = 'temp_image.jpg'
    cv2.imwrite(temp_path, frame)

    # Call your existing face recognition function
    result_image, accuracy, recognized_names = recognize_faces_in_image(temp_path)

    # Log the attendance
    log_attendance(recognized_names)

    # Convert result image to base64-encoded string
    result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()

    # Display the result and accuracy
    return render_template('result.html', result_image=result_image_base64, accuracy=accuracy, recognized_names=recognized_names)

def get_recognized_names(result_image, temp_path):
    # Load the uploaded image
    uploaded_image = face_recognition.load_image_file(temp_path)

    # Detect faces in the uploaded image
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    # Initialize an empty list to store recognized names
    recognized_names = []

    # Loop through each face found in the uploaded image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare each face in the uploaded image with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the matched known face
        if True in matches:
            name = known_names[matches.index(True)]

        # Add the recognized name to the list
        recognized_names.append(name)

    return recognized_names

def log_attendance(recognized_names):
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read the existing attendance log
    try:
        attendance_df = pd.read_excel(attendance_file, engine='openpyxl')
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time"])

    # Loop through recognized names and log the attendance
    for name in recognized_names:
        attendance_df = attendance_df.append({"Name": name, "Time": current_time}, ignore_index=True)

    # Save the updated attendance log
    attendance_df.to_excel(attendance_file, index=False, engine='openpyxl')

def recognize_faces_in_image(image_path):
    # Load the uploaded image
    uploaded_image = face_recognition.load_image_file(image_path)

    # Detect faces in the uploaded image
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    # Load the image to draw rectangles
    result_image = cv2.imread(image_path)

    # Initialize counters for accuracy calculation
    total_faces = 0
    correct_recognitions = 0

    # Initialize an empty list to store recognized names
    recognized_names = []

    # Loop through each face found in the uploaded image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare each face in the uploaded image with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the matched known face
        if True in matches:
            name = known_names[matches.index(True)]

            # If recognized correctly, increase correct recognitions count
            correct_recognitions += 1

        # Draw a rectangle around the face in the result image
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name on the rectangle
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(result_image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Add the recognized name to the list
        recognized_names.append(name)

        # Increase total face count
        total_faces += 1

    # Calculate accuracy
    accuracy = correct_recognitions / total_faces if total_faces > 0 else 0

    return result_image, accuracy, recognized_names

# Create a route for the API endpoint to perform face recognition
@app.route('/api/recognize_faces', methods=['POST'])
def api_recognize_faces():
    try:
        # Get the image file from the request
        file = request.files['file']

        # Save the uploaded file to a temporary location
        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        # Call your existing face recognition function
        result_image, accuracy, recognized_names = recognize_faces_in_image(temp_path)

        # Log the attendance
        log_attendance(recognized_names)

        # Convert result image to base64-encoded string
        result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()

        # Return JSON response
        response_data = {
            "result_image": result_image_base64,
            "accuracy": accuracy,
            "recognized_names": recognized_names
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
