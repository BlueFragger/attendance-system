from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from deepface import DeepFace
import base64
import uuid
import sqlite3
from datetime import datetime, date
import json
import shutil
import gc
import resource  # For memory limiting

app = Flask(__name__)

# Memory optimization - limit memory usage
def limit_memory(max_mem_mb=500):
    """Limit memory usage to help prevent crashes"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_mb * 1024 * 1024, hard))
        print(f"Memory limited to {max_mem_mb}MB")
    except Exception as e:
        print(f"Could not set memory limit: {e}")

# Uncomment to enable memory limiting (adjust based on your Render tier)
# limit_memory(450)

# Database and storage setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'attendance_system.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'faces')
TEMP_FOLDER = os.path.join(BASE_DIR, 'static', 'temp')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        student_id TEXT NOT NULL,
        class TEXT NOT NULL,
        added_date TEXT NOT NULL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        class TEXT NOT NULL,
        check_in_time TEXT NOT NULL,
        date TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        schedule TEXT,
        teacher TEXT
    )
    ''')

    conn.commit()
    conn.close()

# Initialize database
init_db()

class DeepFaceRecognition:
    def __init__(self):
        self.model_name = "VGG-Face"  # Default model, less resource intensive than others
        self.distance_metric = "cosine"
        self.threshold = 0.4  # Threshold for face recognition (lower is better)
        self.detector_backend = "opencv"  # Faster than MTCNN and others
        
    def detect_face(self, image):
        """Detect faces in image with reduced memory usage"""
        try:
            # Use OpenCV's face detector directly instead of DeepFace.extract_faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # If image is a path, read it
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image
                
            # Convert to grayscale for faster processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces - adjust parameters for your needs
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
                
            # Return a list of face objects in DeepFace format for compatibility
            face_objs = []
            for (x, y, w, h) in faces:
                face_obj = {
                    "face": img[y:y+h, x:x+w],
                    "facial_area": {"x": x, "y": y, "w": w, "h": h},
                    "confidence": 0.99  # Default confidence
                }
                face_objs.append(face_obj)
                
            return face_objs
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return None
    
    def add_face_sample(self, image, student_id):
        """Process face image and save it"""
        # Save original image first
        filename = f"{uuid.uuid4()}.jpg"
        img_filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # If image is numpy array, save it
        if isinstance(image, np.ndarray):
            cv2.imwrite(img_filepath, image)
        else:
            # If image was passed as a path
            shutil.copy(image, img_filepath)
            
        # Verify there is a detectable face
        face_objs = self.detect_face(img_filepath)
        
        if not face_objs:
            # Remove the file if no face detected
            if os.path.exists(img_filepath):
                os.remove(img_filepath)
            return None, "No face detected in the image"
            
        # If multiple faces detected, we'll warn but still proceed with the most confident one
        if len(face_objs) > 1:
            print(f"Warning: Multiple faces detected in student {student_id}'s sample. Using the most confident one.")
            
        # Add to database
        relative_img_path = os.path.join('faces', filename)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO face_samples (student_id, image_path) VALUES (?, ?)",
            (student_id, relative_img_path)
        )
        conn.commit()
        conn.close()
        
        return relative_img_path, None
    
    def recognize_face(self, image, class_id=None):
        """Recognize a face with lower memory usage"""
        # Save image to temp location
        temp_img_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.jpg")
        
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_img_path, image)
        else:
            shutil.copy(image, temp_img_path)
            
        # Check if there's any face in the image using lightweight detector
        face_objs = self.detect_face(temp_img_path)
        if not face_objs:
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return None, "No faces detected"
            
        # Get all face samples from database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT fs.student_id, fs.image_path, 
                   s.name, s.student_id as enrollment_id, s.class
            FROM face_samples fs
            JOIN students s ON fs.student_id = s.id
            LIMIT 100  # Add limit to avoid memory issues
        """)
        
        samples = cursor.fetchall()
        conn.close()
        
        if not samples:
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            return {
                "recognized": False,
                "confidence": 0,
                "student": None
            }, None
            
        best_match = None
        best_distance = float('inf')
        match_confidence = 0
        
        # Compare against all samples with batch processing
        batch_size = 10  # Process 10 samples at a time
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            
            for sample in batch:
                sample_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
                
                if not os.path.exists(sample_path):
                    continue
                    
                try:
                    # Lower the size of images before comparison to save memory
                    img1 = cv2.imread(temp_img_path)
                    img2 = cv2.imread(sample_path)
                    
                    if img1 is None or img2 is None:
                        continue
                        
                    # Resize images for faster processing
                    img1 = cv2.resize(img1, (160, 160))
                    img2 = cv2.resize(img2, (160, 160))
                    
                    # Use verify with smaller images
                    result = DeepFace.verify(
                        img1_path=img1,
                        img2_path=img2,
                        model_name=self.model_name,
                        distance_metric=self.distance_metric,
                        detector_backend="skip",  # Skip detection as we already have faces
                        enforce_detection=False,
                        align=False  # Skip alignment to save memory
                    )
                    
                    # Lower distance is better
                    if result["verified"] and result["distance"] < best_distance:
                        best_distance = result["distance"]
                        best_match = dict(sample)
                        # Convert distance to similarity score (0-100%)
                        match_confidence = (1 - min(result["distance"], 1)) * 100
                except Exception as e:
                    print(f"Error comparing faces: {str(e)}")
                    continue
                    
            # Force garbage collection after each batch
            gc.collect()
        
        # Clean up temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        if not best_match:
            return {
                "recognized": False,
                "confidence": 0,
                "student": None
            }, None
            
        # Record attendance
        today = date.today().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")
        student_class = best_match['class'] if class_id is None else class_id
        
        # Check if attendance already recorded today
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE student_id = ? AND date = ? AND class = ?
        """, (best_match['student_id'], today, student_class))
        
        if cursor.fetchone() is None:
            # Record new attendance
            cursor.execute("""
                INSERT INTO attendance (student_id, check_in_time, date, class)
                VALUES (?, ?, ?, ?)
            """, (best_match['student_id'], now, today, student_class))
            attendance_recorded = True
        else:
            attendance_recorded = False
            
        conn.commit()
        conn.close()
        
        # Format response
        student_info = {
            "id": best_match['student_id'],
            "name": best_match['name'],
            "student_id": best_match['enrollment_id'],
            "class": best_match['class'],
            "image": best_match['image_path']
        }
        
        return {
            "recognized": True,
            "confidence": match_confidence,
            "student": student_info,
            "attendance_recorded": attendance_recorded,
            "date": today,
            "time": now
        }, None
        
    def find_face(self, image_path):
        """Find a face in database from single image (used for batch processing)"""
        # Get all face samples from database
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT fs.student_id, fs.image_path, 
                   s.name, s.student_id as enrollment_id, s.class
            FROM face_samples fs
            JOIN students s ON fs.student_id = s.id
        """)
        
        samples = cursor.fetchall()
        conn.close()
        
        if not samples:
            return None
            
        # Use DeepFace.find which is optimized for 1:N matching
        result = DeepFace.find(
            img_path=image_path,
            db_path=UPLOAD_FOLDER,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend=self.detector_backend,
            enforce_detection=False
        )
        
        if result and len(result) > 0 and len(result[0]) > 0:
            # Get the best match
            match = result[0].iloc[0]
            identity = match["identity"]
            
            # Extract student_id from the path
            for sample in samples:
                if sample["image_path"] in identity:
                    return {
                        "student_id": sample["student_id"],
                        "confidence": (1 - match["distance"]) * 100
                    }
        
        return None

# Create global facial recognition system
fr_system = DeepFaceRecognition()

# Simple face comparison fallback
def simple_face_compare(img1, img2):
    """A very simple face comparison that uses much less memory"""
    try:
        # Resize images to same size
        img1 = cv2.resize(img1, (100, 100))
        img2 = cv2.resize(img2, (100, 100))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate simple histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Return true if similarity is high enough
        return {"verified": similarity > 0.5, "distance": 1 - similarity}
    except Exception as e:
        print(f"Simple face compare error: {e}")
        return {"verified": False, "distance": 1.0}

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)



# [All your existing API routes remain the same...]
# Student Management
@app.route('/api/students', methods=['GET', 'POST'])
def manage_students():
    if request.method == 'GET':
        # List all students
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, GROUP_CONCAT(fs.image_path) as face_samples 
            FROM students s
            LEFT JOIN face_samples fs ON s.id = fs.student_id
            GROUP BY s.id
        """)
        students = cursor.fetchall()
        conn.close()
        return jsonify([dict(student) for student in students])
    
    elif request.method == 'POST':
        # Register new student
        data = request.json
        student_id = str(uuid.uuid4())
        
        # Save student info
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (id, name, student_id, class, added_date) VALUES (?, ?, ?, ?, ?)",
            (student_id, data['name'], data['student_id'], data['class'], datetime.now().strftime("%Y-%m-%d"))
        
        # Process face sample
        img_data = data['face_sample'].split(',')[1]  # Remove data URL prefix
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        img_path, error = fr_system.add_face_sample(img, student_id)
        if error:
            conn.rollback()
            conn.close()
            return jsonify({"error": error}), 400
            
        conn.commit()
        conn.close()
        return jsonify({"success": True, "student_id": student_id})

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get face samples to delete
    cursor.execute("SELECT image_path FROM face_samples WHERE student_id = ?", (student_id,))
    samples = cursor.fetchall()
    
    # Delete from database
    cursor.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    
    # Delete face images
    for sample in samples:
        img_path = os.path.join(BASE_DIR, 'static', sample[0])
        if os.path.exists(img_path):
            os.remove(img_path)
    
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# Class Management
@app.route('/api/classes', methods=['GET'])
def get_classes():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classes")
    classes = cursor.fetchall()
    conn.close()
    return jsonify([dict(cls) for cls in classes])

# Attendance Management
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    class_filter = request.args.get('class', '')
    date_filter = request.args.get('date', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT a.*, s.name as student_name, s.student_id, s.class 
        FROM attendance a
        JOIN students s ON a.student_id = s.id
    """
    params = []
    
    conditions = []
    if class_filter:
        conditions.append("s.class = ?")
        params.append(class_filter)
    if date_filter:
        conditions.append("a.date = ?")
        params.append(date_filter)
    if start_date and end_date:
        conditions.append("a.date BETWEEN ? AND ?")
        params.extend([start_date, end_date])
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY a.date DESC, a.check_in_time DESC"
    cursor.execute(query, params)
    attendance = cursor.fetchall()
    conn.close()
    return jsonify([dict(record) for record in attendance])

# Face Recognition
@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    img_data = data['image'].split(',')[1]  # Remove data URL prefix
    img_bytes = base64.b64decode(img_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    class_id = data.get('class', None)
    result, error = fr_system.recognize_face(img, class_id)
    
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain model or update recognition parameters"""
    try:
        # Update model parameters
        if 'model_name' in request.json:
            fr_system.model_name = request.json['model_name']
            
        if 'threshold' in request.json:
            fr_system.threshold = float(request.json['threshold'])
            
        if 'detector_backend' in request.json:
            fr_system.detector_backend = request.json['detector_backend']
            
        # Clear any cached models (this helps with memory issues)
        # Accessing DeepFace's internal functions
        import importlib
        if hasattr(DeepFace.commons, "clear_model_cache"):
            DeepFace.commons.clear_model_cache()
        
        return jsonify({"success": True, "message": "Model parameters updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# [Rest of your existing routes...]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)