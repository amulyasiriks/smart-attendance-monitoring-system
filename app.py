from flask import Flask, request, render_template_string, send_file, redirect, url_for
import os
from detector import detect_faces
from recognizer import Recognizer
from db import init_db, mark_attendance_csv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('demo_data/known_faces', exist_ok=True)

# initialize database & recognizer
init_db('attendance.db')
rec = Recognizer('demo_data/known_faces', use_arcface=False, threshold=0.6)  # Higher threshold for better accuracy

INDEX_HTML = '''
<!doctype html>
<title>Smart Attendance - Demo</title>
<h2>Upload classroom image</h2>
<form method=post enctype=multipart/form-data action="/infer">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p>Sample images available in <code>/sample_images</code> folder inside the project.</p>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/infer', methods=['POST'])
def infer():
    f = request.files.get('file')
    if not f:
        return "No file uploaded", 400
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    # detect
    faces = detect_faces(path)  # list of PIL images (cropped faces)
    results = rec.match_faces(faces)
    # results -> list of (studentid, name, score) or ('unknown', None, score)
    csv_path = mark_attendance_csv(results, source_image=f.filename)
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
