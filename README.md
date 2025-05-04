# Smart Attendance System with Facial Recognition

This is a Flask-based web application that uses facial recognition to mark student attendance.

## Features
- Add and manage students with face samples
- Real-time face recognition using OpenCV
- Attendance tracking and report generation
- SQLite for local dev / PostgreSQL for cloud deployment

## Local Development
```bash
pip install -r requirements.txt
python app.py
```

## Deployment (Render)
1. Push to GitHub
2. Create new Web Service in Render
3. Render auto-detects `render.yaml` and deploys it
