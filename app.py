from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
import base64
import uuid
import sqlite3
from datetime import datetime, date
import traceback
import psycopg2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from urllib.parse import urlparse

app = Flask(__name__)
database_url = os.environ.get('DATABASE_URL', 'sqlite:///attendance_system.db')

# If using Postgres
if database_url.startswith('postgres:'):
    url = urlparse(database_url)
    DB_CONFIG = {
        'user': url.username,
        'password': url.password,
        'host': url.hostname,
        'port': url.port,
        'database': url.path[1:],
    }
    # Use psycopg2 for connections
else:
    # Use SQLite locally
    DB_PATH = 'attendance_system.db'

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

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'faces')
MODEL_PATH = os.path.join(BASE_DIR, 'static', 'model')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
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
        ])
        
        # Use sigmoid for binary classification (1 student) or softmax for multiple
        if num_students == 1:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(num_students, activation='softmax'))
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
        # Store relative path instead of absolute path
        relative_path = os.path.join('faces', filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, face_img_resized)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO face_samples (student_id, image_path) VALUES (?, ?)",
            (student_id, relative_path)  # Store relative path
        )
        conn.commit()
        conn.close()

        return relative_path, None  # Return relative path

    def retrain_model(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM students")
        num_students = max(1, cursor.fetchone()[0])

        if num_students == 0:
            conn.close()
            print("No students in database to train on")
            return {"success": False, "error": "No students in database"}

        # Recreate model if number of students changed
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

        training_data = []
        for student_id, image_path in cursor.fetchall():
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100)) / 255.0
                    training_data.append((img, student_labels[student_id]))

        conn.close()

        if not training_data:
            print("No training data available")
            return {"success": False, "error": "No training data available"}

        # Shuffle training data
        np.random.shuffle(training_data)
        X_train = np.array([x[0] for x in training_data])
        y_train = np.array([x[1] for x in training_data])

        try:
            # Set up data augmentation
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Calculate steps per epoch
            batch_size = 32
            steps = max(1, len(X_train) // batch_size)

            # Train the model
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=10,
                steps_per_epoch=steps,
                verbose=1
            )

            # Save the model
            self.model.save(self.model_path)
            print("Model retrained and saved successfully")

            # Calculate final accuracy
            final_accuracy = history.history['accuracy'][-1] * 100
            print(f"Final training accuracy: {final_accuracy:.2f}%")

            return {
                "success": True,
                "accuracy": final_accuracy,
                "num_samples": len(X_train),
                "num_students": num_students
            }

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def recognize_face(self, image, class_id=None):
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return None, "No faces detected"

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

        face_img = self.preprocess_face(image, largest_face)
        face_img = np.expand_dims(face_img, axis=0)

        conn = sqlite3.connect(DB_PATH)
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
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE student_id = ? AND date = ? AND class = ?
        """, (student[0], today, student[3] if class_id is None else class_id))
        
        if cursor.fetchone() is None:
            # Record new attendance
            cursor.execute("""
                INSERT INTO attendance (student_id, check_in_time, date, class)
                VALUES (?, ?, ?, ?)
            """, (student[0], now, today, student[3] if class_id is None else class_id))
            attendance_recorded = True
        else:
            attendance_recorded = False
            
        conn.commit()
        conn.close()

        return {
            "recognized": True,
            "confidence": confidence,
            "student": {
                "id": student[0],
                "name": student[1],
                "student_id": student[2],
                "class": student[3],
                "image": student[4]
            },
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

    cursor.execute("SELECT image_path FROM face_samples WHERE student_id = ?", (student_id,))
    image_paths = [row[0] for row in cursor.fetchall()]

    cursor.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))

    conn.commit()
    conn.close()

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

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
        # Get training data count first
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM students")
        num_students = cursor.fetchone()[0]
        
        if num_students == 0:
            conn.close()
            return jsonify({
                "success": False,
                "error": "No students in database to train on"
            })
        
        # Get all training samples with absolute paths
        cursor.execute("""
            SELECT fs.student_id, fs.image_path
            FROM face_samples fs
            JOIN students s ON fs.student_id = s.id
        """)
        
        training_data = []
        for student_id, image_path in cursor.fetchall():
            # Convert relative path to absolute path
            abs_path = os.path.join(BASE_DIR, 'static', image_path)
            if os.path.exists(abs_path):
                img = cv2.imread(abs_path)
                if img is not None:
                    img = cv2.resize(img, (100, 100)) / 255.0
                    training_data.append((img, student_id))
        
        conn.close()
        
        if not training_data:
            return jsonify({
                "success": False,
                "error": "No valid training images found"
            })
        
        # Create student to label mapping
        student_ids = sorted(list(set([x[1] for x in training_data])))
        student_to_label = {sid: i for i, sid in enumerate(student_ids)}
        
        # Prepare X and y
        X_train = np.array([x[0] for x in training_data])
        y_train = np.array([student_to_label[x[1]] for x in training_data])
        
        # Recreate model if needed
        if (not hasattr(fr_system.model, 'layers') or 
            fr_system.model.layers[-1].units != len(student_ids)):
            print(f"Recreating model for {len(student_ids)} students")
            fr_system.model = fr_system._create_model()
        
        # Train the model
        batch_size = 32
        steps_per_epoch = max(1, len(X_train) // batch_size)
        
        history = fr_system.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=10,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model
        fr_system.model.save(fr_system.model_path)
        
        return jsonify({
            "success": True,
            "accuracy": float(history.history['accuracy'][-1] * 100),
            "num_samples": len(X_train),
            "num_students": len(student_ids)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

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
    
    # Rest of the function remains the same...
    
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

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)