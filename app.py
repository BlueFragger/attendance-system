from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from datetime import datetime, date, timezone
import pytz  
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
import time  # Added for database retry logic

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key_here'

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

# SQLite connection helper with timeout and retry logic
def get_db_connection(timeout=20.0, isolation_level=None):
    """Get a database connection with timeout and proper isolation level"""
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    if isolation_level is not None:
        conn.isolation_level = isolation_level
    conn.row_factory = sqlite3.Row
    return conn
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        student_id TEXT NOT NULL,
        class TEXT NOT NULL,
        added_date TEXT NOT NULL
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id)
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        class TEXT NOT NULL,
        check_in_time TEXT NOT NULL,
        date TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students (id)
    )''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        schedule TEXT,
        teacher TEXT
    )''')

    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login_user'))
    return render_template('index.html', username=session['user'])

@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user'] = user['username']
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials.")

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_hash = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                           (username, password_hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="Username already exists.")
        conn.close()
        return redirect(url_for('login_user'))

    return render_template('register.html')

@app.route('/logout')
def logout_user():
    session.pop('user', None)
    return redirect(url_for('login_user'))

@app.route('/api/students', methods=['GET'])
def get_students():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(students)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        class_filter = request.args.get('class', '')
        date_filter = request.args.get('date', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')

        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if class_filter:
            query += " AND class = ?"
            params.append(class_filter)

        if date_filter:
            query += " AND date = ?"
            params.append(date_filter)

        if start_date and end_date:
            query += " AND date BETWEEN ? AND ?"
            params.extend([start_date, end_date])

        cursor.execute(query, params)
        attendance = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(attendance)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/students', methods=['POST'])
def register_student():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json

        student_id = str(uuid.uuid4())
        philippines_date = datetime.now(pytz.timezone('Asia/Manila')).strftime("%Y-%m-%d")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (id, name, student_id, class, added_date) VALUES (?, ?, ?, ?, ?)",
            (student_id, data['name'], data['student_id'], data['class'], philippines_date)
        )
        conn.commit()
        conn.close()

        return jsonify({"success": True, "student_id": student_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

def execute_with_retry(query, params=(), max_retries=5, retry_delay=1.0):
    """Execute a database query with retry logic for locks"""
    conn = None
    retries = 0
    
    while retries < max_retries:
        try:
            conn = get_db_connection(timeout=20.0)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            result = cursor.lastrowid
            cursor.close()
            return result
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and retries < max_retries:
                retries += 1
                print(f"Database locked, retrying in {retry_delay}s (attempt {retries}/{max_retries})")
                if conn:
                    conn.close()
                time.sleep(retry_delay)
            else:
                if conn:
                    conn.close()
                raise e
        finally:
            if conn:
                conn.close()
    
    raise sqlite3.OperationalError(f"Database still locked after {max_retries} attempts")

def get_philippines_datetime():
    """Return current date and time in Philippines timezone"""
    manila_tz = pytz.timezone('Asia/Manila')
    return datetime.now(pytz.utc).astimezone(manila_tz)

def init_db():
    conn = get_db_connection()
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
    
    def add_face_sample(self, image, student_id, conn=None):
        """Process face image and save it - now accepts optional connection"""
        # Determine if we should close the connection at the end
        should_close_conn = False
        created_conn = False
        
        try:
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
                
            # Add to database - use the connection passed in or create a new one
            relative_img_path = os.path.join('faces', filename)
            
            # Insert into database with retry logic
            execute_with_retry(
                "INSERT INTO face_samples (student_id, image_path) VALUES (?, ?)",
                (student_id, relative_img_path)
            )
            
            return relative_img_path, None
        except Exception as e:
            print(f"Error adding face sample: {e}")
            # Clean up the file if an error occurred
            if os.path.exists(img_filepath):
                os.remove(img_filepath)
            return None, str(e)
    
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
        conn = get_db_connection()
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
        if class_id is not None and class_id.strip() != "":
            student_class = class_id  # Use the explicit class_id from the request
        else:
            student_class = best_match['class']  # Fall back to student's default class
        
        try:
            # Check if attendance already recorded today
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND class = ?
            """, (best_match['student_id'], today, student_class))
            
            if cursor.fetchone() is None:
                # Record new attendance with retry logic
                execute_with_retry("""
                    INSERT INTO attendance (student_id, check_in_time, date, class)
                    VALUES (?, ?, ?, ?)
                """, (best_match['student_id'], now, today, student_class))
                attendance_recorded = True
            else:
                attendance_recorded = False
                
            conn.close()
        except Exception as e:
            print(f"Error recording attendance: {e}")
            attendance_recorded = False
        
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
        conn = get_db_connection()
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

# API Routes with improved database handling
@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(students)
    except Exception as e:
        print(f"Error getting students: {e}")
        return jsonify({"error": str(e)}), 500

# Student Management
@app.route('/api/students', methods=['POST'])
def register_student():
    try:
        data = request.json
        
        # Check if student with same student_id already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM students WHERE student_id = ?", (data['student_id'],))
        existing_student = cursor.fetchone()
        conn.close()
        
        if existing_student:
            return jsonify({"error": "A student with this ID already exists"}), 400
        
        student_id = str(uuid.uuid4())
        
        # Save student info first with retry logic
        philippines_date = get_philippines_datetime().strftime("%Y-%m-%d")
        execute_with_retry(
            "INSERT INTO students (id, name, student_id, class, added_date) VALUES (?, ?, ?, ?, ?)",
            (student_id, data['name'], data['student_id'], data['class'], philippines_date)
        )
        
        # Process face sample
        img_data = data['face_sample'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        img_path, error = fr_system.add_face_sample(img, student_id)
        if error:
            # If face sample failed, delete the student record
            execute_with_retry("DELETE FROM students WHERE id = ?", (student_id,))
            return jsonify({"error": error}), 400
            
        return jsonify({"success": True, "student_id": student_id})
    except Exception as e:
        print(f"Error registering student: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/students/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get face samples to delete
        cursor.execute("SELECT image_path FROM face_samples WHERE student_id = ?", (student_id,))
        samples = cursor.fetchall()
        conn.close()
        
        # Delete database records with retry logic
        execute_with_retry("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
        execute_with_retry("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        execute_with_retry("DELETE FROM students WHERE id = ?", (student_id,))
        
        # Delete face images
        for sample in samples:
            img_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
            if os.path.exists(img_path):
                os.remove(img_path)
        
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error deleting student: {e}")
        return jsonify({"error": str(e)}), 500

# Class Management
@app.route('/api/classes', methods=['GET'])
def get_classes():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classes")
        classes = [dict(cls) for cls in cursor.fetchall()]
        conn.close()
        return jsonify(classes)
    except Exception as e:
        print(f"Error getting classes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/unique_classes', methods=['GET'])
def get_unique_classes():
    """Get a list of unique classes from registered students"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get unique classes from students table
        cursor.execute("SELECT DISTINCT class FROM students ORDER BY class")
        classes = [row['class'] for row in cursor.fetchall()]
        
        # Also get classes from the classes table if it exists
        cursor.execute("SELECT DISTINCT name FROM classes ORDER BY name")
        class_names = [row['name'] for row in cursor.fetchall()]
        
        # Combine and remove duplicates
        all_classes = list(set(classes + class_names))
        all_classes.sort()  # Sort alphabetically
        
        # Make sure "All Classes" is not in the database results
        if "All Classes" in all_classes:
            all_classes.remove("All Classes")
            
        # Return with "All Classes" as the first option
        result = ["All Classes"] + all_classes
        
        conn.close()
        return jsonify(result)
    except Exception as e:
        print(f"Error getting unique classes: {e}")
        return jsonify(["All Classes"]), 500

# Attendance Management
@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    try:
        class_filter = request.args.get('class', '')
        date_filter = request.args.get('date', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        attendance = get_filtered_attendance(class_filter, date_filter, start_date, end_date)
        return jsonify(attendance)
    except Exception as e:
        print(f"Error getting attendance: {e}")
        return jsonify({"error": str(e)}), 500

def get_filtered_attendance(class_filter=None, date_filter=None, start_date=None, end_date=None):
    """Get filtered attendance records with proper handling of 'All Classes'"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT a.id, s.student_id as student_id, a.class, a.check_in_time, a.date,
                   s.name as student_name, s.id as internal_id, s.class as student_class
            FROM attendance a
            JOIN students s ON a.student_id = s.id
        """
        
        params = []
        conditions = []
        
        # Handle class filter properly
        if class_filter and class_filter.lower() != "all classes":
            conditions.append("a.class = ?")  # Filter on the actual recorded class
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
        attendance = [dict(record) for record in cursor.fetchall()]
        conn.close()
        return attendance
    except Exception as e:
        print(f"Error in get_filtered_attendance: {e}")
        return []

# Face Recognition
@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Wrapper for fixed_recognize_face to maintain backward compatibility"""
    return fixed_recognize_face()

# Add this to your app.py file after the recognize_face route
# Here's the fix for the fixed_recognize_face function
@app.route('/api/fixed_recognize', methods=['POST'])
def fixed_recognize_face():
    """A more memory-efficient face recognition implementation"""
    try:
        data = request.json
        img_data = data['image'].split(',')[1]  # Remove data URL prefix
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        class_id = data.get('class', None)
        
        # Step 1: Detect face in the input image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return jsonify({"error": "No faces detected in image"}), 400
        
        # Extract the first face
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        
        # Step 2: Get all student face samples
        conn = get_db_connection()
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
            return jsonify({
                "recognized": False,
                "confidence": 0,
                "student": None,
                "message": "No students registered in the system"
            })
        
        # Step 3: Compare with each sample using a simple method
        best_match = None
        best_similarity = 0
        
        # Prepare the face for comparison
        face_img = cv2.resize(face_img, (100, 100))
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram for query face
        query_hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
        cv2.normalize(query_hist, query_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Process each sample
        for sample in samples:
            sample_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
            
            if not os.path.exists(sample_path):
                continue
                
            try:
                # Load sample image
                sample_img = cv2.imread(sample_path)
                if sample_img is None:
                    continue
                
                # Find face in sample image
                sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
                sample_faces = face_cascade.detectMultiScale(
                    sample_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                if len(sample_faces) == 0:
                    # If no face detected, use the whole image
                    sample_img_resized = cv2.resize(sample_img, (100, 100))
                    sample_gray_resized = cv2.cvtColor(sample_img_resized, cv2.COLOR_BGR2GRAY)
                else:
                    # Use the first face
                    sx, sy, sw, sh = sample_faces[0]
                    sample_face = sample_img[sy:sy+sh, sx:sx+sw]
                    sample_img_resized = cv2.resize(sample_face, (100, 100))
                    sample_gray_resized = cv2.cvtColor(sample_img_resized, cv2.COLOR_BGR2GRAY)
                
                # Calculate histogram for sample face
                sample_hist = cv2.calcHist([sample_gray_resized], [0], None, [256], [0, 256])
                cv2.normalize(sample_hist, sample_hist, 0, 1, cv2.NORM_MINMAX)
                
                # Compare histograms
                similarity = cv2.compareHist(query_hist, sample_hist, cv2.HISTCMP_CORREL)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = dict(sample)
            except Exception as e:
                print(f"Error comparing with sample {sample['image_path']}: {e}")
                continue
        
        # If no good match found
        if best_match is None or best_similarity < 0.5:  # Adjust threshold as needed
            return jsonify({
                "recognized": False,
                "confidence": best_similarity * 100 if best_match else 0,
                "student": None
            })
        
        # Step 4: Record attendance
        ph_datetime = get_philippines_datetime()
        today = ph_datetime.strftime("%Y-%m-%d")
        now = ph_datetime.strftime("%H:%M:%S")
        
        # FIX HERE: Properly handle class_id
        # If class_id is None, empty string, "All Classes", or "all" (case insensitive), use student's default class
        if class_id is None or class_id.strip() == "" or class_id.lower().strip() in ["all classes", "all"]:
            student_class = best_match['class']  # Use student's registered class
        else:
            student_class = class_id  # Use the explicitly selected class
        
        try:
            # Check if attendance already recorded today
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND class = ?
            """, (best_match['student_id'], today, student_class))
            
            if cursor.fetchone() is None:
                # Record new attendance with retry logic
                execute_with_retry("""
                    INSERT INTO attendance (student_id, check_in_time, date, class)
                    VALUES (?, ?, ?, ?)
                """, (best_match['student_id'], now, today, student_class))
                attendance_recorded = True
            else:
                attendance_recorded = False
                
            conn.close()
        except Exception as e:
            print(f"Error recording attendance: {e}")
            attendance_recorded = False
        
        # Format response
        student_info = {
            "id": best_match['student_id'],
            "name": best_match['name'],
            "student_id": best_match['enrollment_id'],
            "class": best_match['class'],
            "image": best_match['image_path']
        }
        
        return jsonify({
            "recognized": True,
            "confidence": best_similarity * 100,
            "student": student_info,
            "attendance_recorded": attendance_recorded,
            "date": today,
            "time": now,
            "recorded_class": student_class  # Added for debugging purposes
        })
        
    except Exception as e:
        print(f"Error in fixed_recognize_face: {e}")
        return jsonify({"error": str(e)}), 500

# Also fix the debug_recognize_face function with similar logic
@app.route('/api/debug_recognize', methods=['POST'])
def debug_recognize_face():
    """Debug version of recognize_face with better error handling and logging"""
    response = {"debug_info": {}, "error": None}
    
    try:
        # Step 1: Parse the incoming image data
        data = request.json
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        response["debug_info"]["step"] = "Parsing image data"
        img_data = data['image'].split(',')[1]  # Remove data URL prefix
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        # Step 2: Save image to temporary location for debugging
        response["debug_info"]["step"] = "Saving temp image"
        temp_img_path = os.path.join(TEMP_FOLDER, f"debug_{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_img_path, img)
        response["debug_info"]["temp_image_path"] = temp_img_path
        
        # Step 3: Check if we can detect faces in the image
        response["debug_info"]["step"] = "Detecting faces"
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        response["debug_info"]["faces_detected"] = len(faces)
        if len(faces) == 0:
            return jsonify({"error": "No faces detected in image", "debug_info": response["debug_info"]}), 400
            
        # Step 4: Get all students and their face samples
        response["debug_info"]["step"] = "Retrieving students"
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT fs.student_id, fs.image_path, 
                   s.name, s.student_id as enrollment_id, s.class
            FROM face_samples fs
            JOIN students s ON fs.student_id = s.id
        """)
        
        samples = cursor.fetchall()
        conn.close()
        
        response["debug_info"]["samples_count"] = len(samples)
        if len(samples) == 0:
            return jsonify({"error": "No student samples found in database", 
                           "debug_info": response["debug_info"]}), 400
                           
        # Step 5: Try simple comparison first (faster and uses less memory)
        response["debug_info"]["step"] = "Simple face comparison"
        best_match = None
        best_similarity = 0
        
        for sample in samples:
            sample_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
            
            if not os.path.exists(sample_path):
                continue
                
            try:
                sample_img = cv2.imread(sample_path)
                if sample_img is None:
                    continue
                    
                # Extract face from both images for better comparison
                for (x, y, w, h) in faces:
                    face_img = img[y:y+h, x:x+w]
                    
                    # Find faces in sample image
                    sample_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
                    sample_faces = face_cascade.detectMultiScale(
                        sample_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    if len(sample_faces) == 0:
                        continue
                        
                    # Compare with first face found
                    sx, sy, sw, sh = sample_faces[0]
                    sample_face = sample_img[sy:sy+sh, sx:sx+sw]
                    
                    # Simple comparison
                    face_img = cv2.resize(face_img, (100, 100))
                    sample_face = cv2.resize(sample_face, (100, 100))
                    
                    # Convert to grayscale for histogram comparison
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    sample_face_gray = cv2.cvtColor(sample_face, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate histograms
                    hist1 = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([sample_face_gray], [0], None, [256], [0, 256])
                    
                    # Normalize histograms
                    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
                    
                    # Compare histograms
                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = dict(sample)
            except Exception as e:
                response["debug_info"]["comparison_errors"] = str(e)
                continue
        
        # If we found a match with simple comparison
        if best_match and best_similarity > 0.6:  # 0.6 threshold can be adjusted
            response["debug_info"]["recognition_method"] = "simple_histogram"
            response["debug_info"]["similarity"] = best_similarity
            
            # Record attendance
            class_id = data.get('class', None)
            
            # FIX HERE: Properly handle class_id
            # If class_id is None, empty string, "All Classes", or "all" (case insensitive), use student's default class
            if class_id is None or class_id.strip() == "" or class_id.lower().strip() in ["all classes", "all"]:
                student_class = best_match['class']  # Use student's registered class
            else:
                student_class = class_id  # Use the explicitly selected class
            
            response["debug_info"]["class_value"] = class_id
            response["debug_info"]["student_class"] = student_class

            today = date.today().strftime("%Y-%m-%d")
            now = datetime.now().strftime("%H:%M:%S")
            
            # Check if attendance already recorded today
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND class = ?
            """, (best_match['student_id'], today, student_class))
            
            if cursor.fetchone() is None:
                execute_with_retry("""
                    INSERT INTO attendance (student_id, check_in_time, date, class)
                    VALUES (?, ?, ?, ?)
                """, (best_match['student_id'], now, today, student_class))
                attendance_recorded = True
            else:
                attendance_recorded = False
                
            conn.close()
            
            # Format response
            student_info = {
                "id": best_match['student_id'],
                "name": best_match['name'],
                "student_id": best_match['enrollment_id'],
                "class": best_match['class'],
                "image": best_match['image_path']
            }
            
            result = {
                "recognized": True,
                "confidence": best_similarity * 100,
                "student": student_info,
                "attendance_recorded": attendance_recorded,
                "date": today,
                "time": now,
                "recorded_class": student_class,  # Added for debugging
                "debug_info": response["debug_info"]
            }
            
            return jsonify(result)
        
        # Step 6: If we didn't find a good match with simple comparison,
        # try a more robust approach
        response["debug_info"]["step"] = "Advanced face recognition"
        response["debug_info"]["recognition_method"] = "deepface_fallback"
        
        try:
            # Only extract the first face for recognition
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_path = os.path.join(TEMP_FOLDER, f"face_{uuid.uuid4()}.jpg")
                cv2.imwrite(face_path, face_img)
                
                # Try DeepFace with reduced parameters
                try:
                    best_match = None
                    best_distance = float('inf')
                    match_confidence = 0
                    
                    for sample in samples:
                        sample_path = os.path.join(BASE_DIR, 'static', sample['image_path'])
                        
                        if not os.path.exists(sample_path):
                            continue
                            
                        try:
                            result = DeepFace.verify(
                                img1_path=face_path,
                                img2_path=sample_path,
                                model_name="VGG-Face",  # Simpler model
                                distance_metric="cosine",
                                detector_backend="skip",  # Skip detection since we already have the face
                                enforce_detection=False,
                                align=False  # Skip alignment to save memory
                            )
                            
                            if result["verified"] and result["distance"] < best_distance:
                                best_distance = result["distance"]
                                best_match = dict(sample)
                                match_confidence = (1 - min(result["distance"], 1)) * 100
                        except Exception as e:
                            continue
                            
                    if not best_match:
                        response["debug_info"]["deepface_result"] = "No match found"
                        return jsonify({"recognized": False, "confidence": 0, "student": None, 
                                       "debug_info": response["debug_info"]})
                    
                    # Record attendance
                    class_id = data.get('class', None)
                    
                    # FIX HERE: Properly handle class_id
                    # If class_id is None, empty string, "All Classes", or "all" (case insensitive), use student's default class
                    if class_id is None or class_id.strip() == "" or class_id.lower().strip() in ["all classes", "all"]:
                        student_class = best_match['class']  # Use student's registered class
                    else:
                        student_class = class_id  # Use the explicitly selected class
                        
                    response["debug_info"]["class_value"] = class_id
                    response["debug_info"]["student_class"] = student_class
                    
                    today = date.today().strftime("%Y-%m-%d")
                    now = datetime.now().strftime("%H:%M:%S")
                    
                    # Check if attendance already recorded today
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id FROM attendance 
                        WHERE student_id = ? AND date = ? AND class = ?
                    """, (best_match['student_id'], today, student_class))
                    
                    if cursor.fetchone() is None:
                        execute_with_retry("""
                            INSERT INTO attendance (student_id, check_in_time, date, class)
                            VALUES (?, ?, ?, ?)
                        """, (best_match['student_id'], now, today, student_class))
                        attendance_recorded = True
                    else:
                        attendance_recorded = False
                        
                    conn.close()
                    
                    # Format response
                    student_info = {
                        "id": best_match['student_id'],
                        "name": best_match['name'],
                        "student_id": best_match['enrollment_id'],
                        "class": best_match['class'],
                        "image": best_match['image_path']
                    }
                    
                    result = {
                        "recognized": True,
                        "confidence": match_confidence,
                        "student": student_info,
                        "attendance_recorded": attendance_recorded,
                        "date": today,
                        "time": now,
                        "recorded_class": student_class,  # Added for debugging
                        "debug_info": response["debug_info"]
                    }
                    
                    # Clean up temp files
                    if os.path.exists(face_path):
                        os.remove(face_path)
                    
                    return jsonify(result)
                    
                except Exception as e:
                    response["debug_info"]["deepface_error"] = str(e)
                    # If DeepFace failed, just clean up and continue
                    if os.path.exists(face_path):
                        os.remove(face_path)
                
                # Clean up temp files
                if os.path.exists(face_path):
                    os.remove(face_path)
                
                # Break after processing first face
                break
                
        except Exception as e:
            response["debug_info"]["face_extraction_error"] = str(e)
        
        # At this point, all recognition methods have failed
        return jsonify({
            "recognized": False,
            "confidence": 0,
            "student": None,
            "debug_info": response["debug_info"]
        })
        
    except Exception as e:
        # Catch all other errors
        print(f"Debug recognize error: {str(e)}")
        response["error"] = str(e)
        return jsonify({"error": str(e), "debug_info": response["debug_info"]}), 500
    finally:
        # Clean up temp image
        if 'temp_img_path' in response["debug_info"] and os.path.exists(response["debug_info"]["temp_img_path"]):
            os.remove(response["debug_info"]["temp_img_path"])
            
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
