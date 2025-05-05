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

app = Flask(__name__)

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
        """Detect faces in image and return face locations"""
        try:
            # DeepFace detect will extract face regions
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if not face_objs or len(face_objs) == 0:
                return None
                
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
        """Recognize a face in the image"""
        # Save image to temp location
        temp_img_path = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.jpg")
        
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_img_path, image)
        else:
            shutil.copy(image, temp_img_path)
            
        # Check if there's any face in the image
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
        
        # Compare against all samples
        for sample in samples:
            sample_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
            
            if not os.path.exists(sample_path):
                continue
                
            try:
                # Use verify instead of find as we're comparing one-to-one
                result = DeepFace.verify(
                    img1_path=temp_img_path,
                    img2_path=sample_path,
                    model_name=self.model_name,
                    distance_metric=self.distance_metric,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
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

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.id, s.name, s.student_id, s.class, s.added_date,
               (SELECT image_path FROM face_samples WHERE student_id = s.id LIMIT 1) as image,
               COUNT(fs.id) as sample_count
        FROM students s
        LEFT JOIN face_samples fs ON s.id = fs.student_id
        GROUP BY s.id
    """)

    students = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify(students)

@app.route('/api/students/<student_id>', methods=['GET'])
def get_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.id, s.name, s.student_id, s.class, s.added_date
        FROM students s
        WHERE s.id = ?
    """, (student_id,))

    student = cursor.fetchone()
    if not student:
        conn.close()
        return jsonify({"error": "Student not found"}), 404

    student_data = dict(student)

    cursor.execute("""
        SELECT id, image_path
        FROM face_samples
        WHERE student_id = ?
    """, (student_id,))

    student_data['samples'] = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify(student_data)

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.json

    if not data or 'name' not in data or 'student_id' not in data or 'class' not in data:
        return jsonify({"error": "Name, student ID, and class are required"}), 400

    student_uuid = str(uuid.uuid4())[:8]
    added_date = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO students (id, name, student_id, class, added_date) VALUES (?, ?, ?, ?, ?)",
        (student_uuid, data['name'], data['student_id'], data['class'], added_date)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "id": student_uuid,
        "name": data['name'],
        "student_id": data['student_id'],
        "class": data['class'],
        "added_date": added_date
    })

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get face sample paths
    cursor.execute("SELECT image_path FROM face_samples WHERE student_id = ?", (student_id,))
    sample_paths = cursor.fetchall()

    # Delete database records
    cursor.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))

    conn.commit()
    conn.close()

    # Delete files
    for (image_path,) in sample_paths:
        full_path = os.path.join(BASE_DIR, 'static', image_path)
        if os.path.exists(full_path):
            os.remove(full_path)

    return jsonify({"success": True})

@app.route('/api/face_samples', methods=['POST'])
def add_face_sample():
    if 'image' not in request.json or 'student_id' not in request.json:
        return jsonify({"error": "Image and student_id are required"}), 400

    try:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        filepath, error = fr_system.add_face_sample(image, request.json['student_id'])
        if error:
            return jsonify({"error": error}), 400

        return jsonify({
            "success": True,
            "filepath": filepath
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.json:
        return jsonify({"error": "Image is required"}), 400

    class_id = request.json.get('class_id')

    try:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result, error = fr_system.recognize_face(image, class_id)

        if error:
            return jsonify({"error": error}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM classes")
    classes = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return jsonify(classes)

@app.route('/api/classes', methods=['POST'])
def add_class():
    data = request.json

    if not data or 'name' not in data:
        return jsonify({"error": "Class name is required"}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO classes (name, schedule, teacher) VALUES (?, ?, ?)",
        (data['name'], data.get('schedule', ''), data.get('teacher', ''))
    )
    class_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "id": class_id,
        "name": data['name'],
        "schedule": data.get('schedule', ''),
        "teacher": data.get('teacher', '')
    })

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    date_param = request.args.get('date', date.today().strftime("%Y-%m-%d"))
    class_param = request.args.get('class')
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
        SELECT a.id, a.student_id, a.check_in_time, a.date, a.class,
               s.name, s.student_id as student_code
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
    """
    params = [date_param]
    
    if class_param:
        query += " AND a.class = ?"
        params.append(class_param)
        
    cursor.execute(query, params)
    attendance_records = [dict(row) for row in cursor.fetchall()]
    
    # Get all students in the class for reporting absences
    if class_param:
        cursor.execute("""
            SELECT id, name, student_id FROM students 
            WHERE class = ?
        """, (class_param,))
        all_students = [dict(row) for row in cursor.fetchall()]
        
        # Mark present/absent
        present_ids = [record['student_id'] for record in attendance_records]
        for student in all_students:
            student['present'] = student['id'] in present_ids
            
        attendance_data = {
            'date': date_param,
            'class': class_param,
            'attendance_records': attendance_records,
            'all_students': all_students,
            'present_count': len(present_ids),
            'absent_count': len(all_students) - len(present_ids)
        }
    else:
        attendance_data = {
            'date': date_param,
            'attendance_records': attendance_records
        }
    
    conn.close()
    return jsonify(attendance_data)

@app.route('/api/attendance/report', methods=['GET'])
def attendance_report():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date', date.today().strftime("%Y-%m-%d"))
    class_param = request.args.get('class')
    student_id = request.args.get('student_id')
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query_parts = ["""
        SELECT a.date, a.class,
               COUNT(DISTINCT a.student_id) as present_count,
               (SELECT COUNT(*) FROM students WHERE class = a.class) as total_students
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE 1=1
    """]
    params = []
    
    if start_date:
        query_parts.append("AND a.date >= ?")
        params.append(start_date)
        
    if end_date:
        query_parts.append("AND a.date <= ?")
        params.append(end_date)
        
    if class_param:
        query_parts.append("AND a.class = ?")
        params.append(class_param)
        
    if student_id:
        query_parts.append("AND s.student_id = ?")
        params.append(student_id)
        
    query_parts.append("GROUP BY a.date, a.class")
    query = " ".join(query_parts)
    
    cursor.execute(query, params)
    report_data = [dict(row) for row in cursor.fetchall()]
    
    # Calculate attendance percentages
    for record in report_data:
        if record['total_students'] > 0:
            record['attendance_rate'] = round((record['present_count'] / record['total_students']) * 100, 1)
        else:
            record['attendance_rate'] = 0
            
    conn.close()
    return jsonify({
        'start_date': start_date,
        'end_date': end_date,
        'class': class_param,
        'student_id': student_id,
        'report_data': report_data
    })

@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    """Process a batch of images for attendance"""
    if 'images' not in request.json or not isinstance(request.json['images'], list):
        return jsonify({"error": "Images array is required"}), 400
        
    class_id = request.json.get('class_id')
    if not class_id:
        return jsonify({"error": "Class ID is required for batch processing"}), 400
        
    results = []
    recognized_students = set()
    
    for img_data in request.json['images']:
        try:
            # Process each image
            image_data = img_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Save to temp file
            temp_file = os.path.join(TEMP_FOLDER, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(temp_file, image)
            
            # Find faces
            try:
                face_info = fr_system.find_face(temp_file)
                
                if face_info and face_info["student_id"] not in recognized_students:
                    # Mark attendance
                    today = date.today().strftime("%Y-%m-%d")
                    now = datetime.now().strftime("%H:%M:%S")
                    
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    # Get student details
                    cursor.execute("""
                        SELECT id, name, student_id, class FROM students
                        WHERE id = ?
                    """, (face_info["student_id"],))
                    
                    student = cursor.fetchone()
                    
                    if student:
                        # Check if attendance already recorded
                        cursor.execute("""
                            SELECT id FROM attendance 
                            WHERE student_id = ? AND date = ? AND class = ?
                        """, (student['id'], today, class_id))
                        
                        if cursor.fetchone() is None:
                            # Record new attendance
                            cursor.execute("""
                                INSERT INTO attendance (student_id, check_in_time, date, class)
                                VALUES (?, ?, ?, ?)
                            """, (student['id'], now, today, class_id))
                            
                            recognized_students.add(face_info["student_id"])
                            results.append({
                                "recognized": True,
                                "student": {
                                    "id": student['id'],
                                    "name": student['name'],
                                    "student_id": student['student_id'],
                                    "class": student['class']
                                },
                                "confidence": face_info["confidence"]
                            })
                    
                    conn.commit()
                    conn.close()
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            continue
    
    return jsonify({
        "success": True,
        "recognized_count": len(recognized_students),
        "results": results
    })

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get information about the current model configuration"""
    return jsonify({
        "model_name": fr_system.model_name,
        "distance_metric": fr_system.distance_metric,
        "detector_backend": fr_system.detector_backend,
        "threshold": fr_system.threshold
    })

@app.route('/api/model_config', methods=['POST'])
def configure_model():
    """Update model configuration"""
    data = request.json
    
    if 'model_name' in data:
        valid_models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace", "SFace"]
        if data['model_name'] in valid_models:
            fr_system.model_name = data['model_name']
    
    if 'distance_metric' in data:
        valid_metrics = ["cosine", "euclidean", "euclidean_l2"]
        if data['distance_metric'] in valid_metrics:
            fr_system.distance_metric = data['distance_metric']
    
    if 'detector_backend' in data:
        valid_backends = ["opencv", "ssd", "mtcnn", "retinaface"]
        if data['detector_backend'] in valid_backends:
            fr_system.detector_backend = data['detector_backend']
    
    if 'threshold' in data:
        threshold = float(data['threshold'])
        if 0 < threshold < 1:
            fr_system.threshold = threshold
    
    return jsonify({
        "success": True,
        "model_name": fr_system.model_name,
        "distance_metric": fr_system.distance_metric,
        "detector_backend": fr_system.detector_backend,
        "threshold": fr_system.threshold
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
