from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
import base64
import uuid
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, date
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from urllib.parse import urlparse

app = Flask(__name__)

# Database connection setup
def get_db_connection():
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///attendance_system.db')
    
    if database_url.startswith('postgres:'):
        # Handle Heroku's older postgres:// URLs (they now use postgresql://)
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        url = urlparse(database_url)
        conn = psycopg2.connect(
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port,
            database=url.path[1:],
        )
        conn.autocommit = True
        return conn
    else:
        # For local development, fall back to SQLite
        import sqlite3
        conn = sqlite3.connect('attendance_system.db')
        conn.row_factory = sqlite3.Row
        return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we're using PostgreSQL
    if isinstance(conn, psycopg2.extensions.connection):
        # PostgreSQL schema
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
            id SERIAL PRIMARY KEY,
            student_id TEXT NOT NULL REFERENCES students(id),
            image_path TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            student_id TEXT NOT NULL REFERENCES students(id),
            class TEXT NOT NULL,
            check_in_time TEXT NOT NULL,
            date TEXT NOT NULL
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            schedule TEXT,
            teacher TEXT
        )
        ''')
    else:
        # SQLite schema
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

# Call init_db to setup tables
init_db()

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'faces')
MODEL_PATH = os.path.join(BASE_DIR, 'static', 'model')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# For Heroku, we'll store image paths relative to BASE_DIR
def get_relative_path(absolute_path):
    if absolute_path and BASE_DIR in absolute_path:
        return os.path.relpath(absolute_path, BASE_DIR)
    return absolute_path

def get_absolute_path(relative_path):
    if relative_path:
        return os.path.join(BASE_DIR, relative_path)
    return relative_path

class FacialRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model_path = os.path.join(MODEL_PATH, "facial_recognition_model.h5")

        if os.path.exists(self.model_path):
            print("Loading existing facial recognition model...")
            self.model = load_model(self.model_path)
        else:
            print("Creating new facial recognition model...")
            self.model = self._create_model()

    def _create_model(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if isinstance(conn, psycopg2.extensions.connection):
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = max(1, cursor.fetchone()[0])
        else:
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = max(1, cursor.fetchone()[0])
            
        conn.close()

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_students, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def preprocess_face(self, image, face):
        x, y, w, h = face
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_img = face_img / 255.0
        return face_img

    def add_face_sample(self, image, student_id):
        faces = self.detect_faces(image)

        if len(faces) != 1:
            return None, "Expected exactly one face in the image"

        x, y, w, h = faces[0]
        face_img = image[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (100, 100))

        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, face_img_resized)
        
        # Store relative path for database
        relative_filepath = get_relative_path(filepath)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        if isinstance(conn, psycopg2.extensions.connection):
            cursor.execute(
                "INSERT INTO face_samples (student_id, image_path) VALUES (%s, %s)",
                (student_id, relative_filepath)
            )
        else:
            cursor.execute(
                "INSERT INTO face_samples (student_id, image_path) VALUES (?, ?)",
                (student_id, relative_filepath)
            )
            
        conn.commit()
        conn.close()

        return filepath, None

    def retrain_model(self):
        conn = get_db_connection()
        
        if isinstance(conn, psycopg2.extensions.connection):
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = max(1, cursor.fetchone()[0])
            
            if self.model.layers[-1].units != num_students:
                print(f"Recreating model for {num_students} students")
                self.model = self._create_model()

            X_train = []
            y_train = []

            cursor.execute("SELECT id, row_number() OVER (ORDER BY id) - 1 AS label FROM students")
            student_labels = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT fs.student_id, fs.image_path
                FROM face_samples fs
                JOIN students s ON fs.student_id = s.id
            """)
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = max(1, cursor.fetchone()[0])
            
            if self.model.layers[-1].units != num_students:
                print(f"Recreating model for {num_students} students")
                self.model = self._create_model()

            X_train = []
            y_train = []

            cursor.execute("SELECT id, rowid-1 AS label FROM students")
            student_labels = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("""
                SELECT fs.student_id, fs.image_path
                FROM face_samples fs
                JOIN students s ON fs.student_id = s.id
            """)

        for row in cursor.fetchall():
            student_id, image_path = row
            abs_image_path = get_absolute_path(image_path)
            
            if os.path.exists(abs_image_path):
                img = cv2.imread(abs_image_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100)) / 255.0
                    X_train.append(img)
                    y_train.append(student_labels[student_id])

        conn.close()

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if len(X_train) > 0:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=32),
                epochs=10,
                steps_per_epoch=len(X_train) // 32 + 1
            )

            self.model.save(self.model_path)
            print("Model retrained and saved successfully")
            return True
        else:
            print("No training data available")
            return False

    def recognize_face(self, image, class_id=None):
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None, "No faces detected"

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

        face_img = self.preprocess_face(image, largest_face)
        face_img = np.expand_dims(face_img, axis=0)

        conn = get_db_connection()
        
        if isinstance(conn, psycopg2.extensions.connection):
            cursor = conn.cursor(cursor_factory=DictCursor)
            
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = cursor.fetchone()[0]

            if num_students == 0:
                conn.close()
                return None, "No students in database"

            predictions = self.model.predict(face_img, verbose=0)
            student_label = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)

            cursor.execute("""
                SELECT s.id, s.name, s.student_id, s.class,
                       (SELECT image_path FROM face_samples WHERE student_id = s.id LIMIT 1) as image
                FROM students s
                WHERE row_number() OVER (ORDER BY id) - 1 = %s
            """, (student_label,))
        else:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM students")
            num_students = cursor.fetchone()[0]

            if num_students == 0:
                conn.close()
                return None, "No students in database"

            predictions = self.model.predict(face_img, verbose=0)
            student_label = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]) * 100)

            cursor.execute("""
                SELECT s.id, s.name, s.student_id, s.class,
                       (SELECT image_path FROM face_samples WHERE student_id = s.id LIMIT 1) as image
                FROM students s
                WHERE rowid-1 = ?
            """, (student_label,))

        student = cursor.fetchone()
        
        if not student or confidence < 60:
            conn.close()
            return {
                "recognized": False,
                "confidence": confidence,
                "student": None
            }, None

        # Record attendance
        today = date.today().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M:%S")
        
        # Check if attendance already recorded today
        if isinstance(conn, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = %s AND date = %s AND class = %s
            """, (student['id'], today, student['class'] if class_id is None else class_id))
            
            if cursor.fetchone() is None:
                # Record new attendance
                cursor.execute("""
                    INSERT INTO attendance (student_id, check_in_time, date, class)
                    VALUES (%s, %s, %s, %s)
                """, (student['id'], now, today, student['class'] if class_id is None else class_id))
                attendance_recorded = True
            else:
                attendance_recorded = False
        else:
            cursor.execute("""
                SELECT id FROM attendance 
                WHERE student_id = ? AND date = ? AND class = ?
            """, (student['id'], today, student['class'] if class_id is None else class_id))
            
            if cursor.fetchone() is None:
                # Record new attendance
                cursor.execute("""
                    INSERT INTO attendance (student_id, check_in_time, date, class)
                    VALUES (?, ?, ?, ?)
                """, (student['id'], now, today, student['class'] if class_id is None else class_id))
                attendance_recorded = True
            else:
                attendance_recorded = False
                
        conn.commit()
        
        # Convert to dict for JSON serialization
        if isinstance(conn, psycopg2.extensions.connection):
            student_dict = dict(student)
        else:
            student_dict = {
                "id": student[0],
                "name": student[1],
                "student_id": student[2],
                "class": student[3],
                "image": student[4]
            }
            
        # Convert relative path to absolute for frontend
        if student_dict["image"]:
            student_dict["image"] = get_absolute_path(student_dict["image"])
            
        conn.close()

        return {
            "recognized": True,
            "confidence": confidence,
            "student": student_dict,
            "attendance_recorded": attendance_recorded,
            "date": today,
            "time": now
        }, None

# Create global facial recognition system
fr_system = FacialRecognitionSystem()

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = get_db_connection()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
            SELECT s.id, s.name, s.student_id, s.class, s.added_date,
                (SELECT image_path FROM face_samples WHERE student_id = s.id LIMIT 1) as image,
                COUNT(fs.id) as sample_count
            FROM students s
            LEFT JOIN face_samples fs ON s.id = fs.student_id
            GROUP BY s.id
        """)
        
        students = [dict(row) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.student_id, s.class, s.added_date,
                (SELECT image_path FROM face_samples WHERE student_id = s.id LIMIT 1) as image,
                COUNT(fs.id) as sample_count
            FROM students s
            LEFT JOIN face_samples fs ON s.id = fs.student_id
            GROUP BY s.id
        """)
        
        columns = [column[0] for column in cursor.description]
        students = []
        for row in cursor.fetchall():
            student = dict(zip(columns, row))
            students.append(student)
    
    # Convert relative paths to absolute for frontend
    for student in students:
        if student["image"]:
            student["image"] = get_absolute_path(student["image"])
    
    conn.close()
    return jsonify(students)

@app.route('/api/students/<student_id>', methods=['GET'])
def get_student(student_id):
    conn = get_db_connection()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        cursor.execute("""
            SELECT s.id, s.name, s.student_id, s.class, s.added_date
            FROM students s
            WHERE s.id = %s
        """, (student_id,))
        
        student = cursor.fetchone()
        if not student:
            conn.close()
            return jsonify({"error": "Student not found"}), 404

        student_data = dict(student)

        cursor.execute("""
            SELECT id, image_path
            FROM face_samples
            WHERE student_id = %s
        """, (student_id,))
        
        student_data['samples'] = [dict(row) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.id, s.name, s.student_id, s.class, s.added_date
            FROM students s
            WHERE s.id = ?
        """, (student_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Student not found"}), 404
            
        columns = [column[0] for column in cursor.description]
        student_data = dict(zip(columns, row))

        cursor.execute("""
            SELECT id, image_path
            FROM face_samples
            WHERE student_id = ?
        """, (student_id,))
        
        samples_columns = [column[0] for column in cursor.description]
        student_data['samples'] = [dict(zip(samples_columns, row)) for row in cursor.fetchall()]
    
    # Convert relative paths to absolute for frontend
    for sample in student_data['samples']:
        if sample["image_path"]:
            sample["image_path"] = get_absolute_path(sample["image_path"])
    
    conn.close()
    return jsonify(student_data)

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.json

    if not data or 'name' not in data or 'student_id' not in data or 'class' not in data:
        return jsonify({"error": "Name, student ID, and class are required"}), 400

    student_uuid = str(uuid.uuid4())[:8]
    added_date = datetime.now().strftime("%Y-%m-%d")

    conn = get_db_connection()
    cursor = conn.cursor()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor.execute(
            "INSERT INTO students (id, name, student_id, class, added_date) VALUES (%s, %s, %s, %s, %s)",
            (student_uuid, data['name'], data['student_id'], data['class'], added_date)
        )
    else:
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
    conn = get_db_connection()
    cursor = conn.cursor()

    if isinstance(conn, psycopg2.extensions.connection):
        cursor.execute("SELECT image_path FROM face_samples WHERE student_id = %s", (student_id,))
    else:
        cursor.execute("SELECT image_path FROM face_samples WHERE student_id = ?", (student_id,))
        
    image_paths = [row[0] for row in cursor.fetchall()]

    if isinstance(conn, psycopg2.extensions.connection):
        cursor.execute("DELETE FROM face_samples WHERE student_id = %s", (student_id,))
        cursor.execute("DELETE FROM attendance WHERE student_id = %s", (student_id,))
        cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))
    else:
        cursor.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))

    conn.commit()
    conn.close()

    for path in image_paths:
        abs_path = get_absolute_path(path)
        if os.path.exists(abs_path):
            os.remove(abs_path)

    fr_system.retrain_model()

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

@app.route('/api/retrain', methods=['POST'])
def retrain():
    try:
        success = fr_system.retrain_model()
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    conn = get_db_connection()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute("SELECT * FROM classes")
        classes = [dict(row) for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM classes")
        columns = [column[0] for column in cursor.description]
        classes = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    conn.close()
    return jsonify(classes)

@app.route('/api/classes', methods=['POST'])
def add_class():
    data = request.json

    if not data or 'name' not in data:
        return jsonify({"error": "Class name is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor.execute(
            "INSERT INTO classes (name, schedule, teacher) VALUES (%s, %s, %s) RETURNING id",
            (data['name'], data.get('schedule', ''), data.get('teacher', ''))
        )
        class_id = cursor.fetchone()[0]
    else:
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
    
    conn = get_db_connection()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        query = """
            SELECT a.id, a.student_id, a.check_in_time, a.date, a.class,
                s.name, s.student_id as student_code
            FROM attendance a
            JOIN students s ON a.student_id = s.id
            WHERE a.date = %s
        """
        params = [date_param]
        
        if class_param:
            query += " AND a.class = %s"
            params.append(class_param)
            
        cursor.execute(query, params)
        attendance_records = [dict(row) for row in cursor.fetchall()]
    else:
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
        columns = [column[0] for column in cursor.description]
        attendance_records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    # Get all students in the class for reporting absences
    if class_param:
        if isinstance(conn, psycopg2.extensions.connection):
            cursor.execute("""
                SELECT id, name, student_id FROM students 
                WHERE class = %s
            """, (class_param,))
            all_students = [dict(row) for row in cursor.fetchall()]
        else:
            cursor.execute("""
                SELECT id, name, student_id FROM students 
                WHERE class = ?
            """, (class_param,))
            columns = [column[0] for column in cursor.description]
            all_students = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
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
    
    conn = get_db_connection()
    
    if isinstance(conn, psycopg2.extensions.connection):
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        query_parts = ["""
            SELECT a.date, a.class,
                COUNT(DISTINCT a.student_id) as present_count,
                (SELECT COUNT(*) FROM students WHERE class = a.class)