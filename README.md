# Smart Attendance Monitoring — Prototype (Runnable)

This is a runnable prototype of the Smart Attendance Monitoring project you uploaded.
It uses a simple, reliable stack so you can run it locally out-of-the-box:

- Face detection & embeddings: **facenet-pytorch** (MTCNN + InceptionResnetV1)
- Web UI & API: **Flask**
- Storage: **SQLite** (attendance.db)
- Export: CSV (attendance.csv)

Notes:
- Your original design mentioned YOLOv8 for multi-face detection and ArcFace for recognition.
  This project is intentionally shipped using **facenet-pytorch** so it runs easily on CPU.
  The code is modular: see `detector.py` and `recognizer.py` for where to plug YOLOv8 / ArcFace.
- To switch to YOLOv8 or ArcFace, follow the comments in the files and the CLI steps in this README.

## What's included
- `app.py` — Flask app with a simple upload form and `/infer` endpoint
- `detector.py` — detection abstraction (defaults to MTCNN)
- `recognizer.py` — embedding + simple matching (defaults to InceptionResnetV1)
- `db.py` — simple SQLite helper (creates attendance table)
- `requirements.txt`
- `sample_images/` — two small placeholder images so app works immediately
- `run_app.sh` — helper to create venv and run the app
- `demo_data/known_faces/` — folder where you can add labeled images (format: <id>_<name>.jpg)

## Quick start (Linux / macOS)
1. Open terminal in this folder.
2. Create virtualenv and install:
   ```
   bash run_app.sh
   ```
   The script will create a `venv`, install packages, and start the Flask app on port 5000.
3. Open `http://127.0.0.1:5000` in your browser.
4. Upload a classroom image (or try the provided sample images). After processing you'll get a CSV of attendance.

## How matching works (simple default)
- The default recognizer builds embeddings of images inside `demo_data/known_faces/`.
- File naming format for known faces: `<studentid>_<FullName>.jpg` (e.g. `1001_JohnDoe.jpg`)
- On inference, each detected face is embedded and compared with known embeddings using cosine similarity.
- Threshold is configurable in `recognizer.py`.

## To move to YOLOv8 + ArcFace (notes)
- Install `ultralytics` for YOLOv8 detection and replace `detector.py`'s MTCNN calls with YOLO inference.
- Install `insightface` (ArcFace) models for embeddings. Replace the embedding code in `recognizer.py`.
- Both are documented in comments inside those files.

## Limitations & next steps
- This is a prototype meant to run on CPU. For real-time multi-face detection of 60–70 students
  in a single frame you should use GPU, YOLOv8 (or RetinaFace) for fast detection and ArcFace
  for robust embeddings. Also add batch processing and caching for performance.

