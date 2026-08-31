import sqlite3
import csv
from datetime import datetime
import os

def init_db(db_path='attendance.db'):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            student_name TEXT,
            score REAL,
            timestamp TEXT,
            source_image TEXT
        )
    ''')
    conn.commit()
    conn.close()

def mark_attendance_csv(results, db_path='attendance.db', source_image='unknown.jpg'):
    '''
    results: list of tuples (studentid_or_unknown, name_or_None, score)
    Returns path to CSV file created.
    '''
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    rows = []
    for sid, name, score in results:
        cur.execute('INSERT INTO attendance (student_id, student_name, score, timestamp, source_image) VALUES (?,?,?,?,?)',
                    (sid, name, float(score), now, source_image))
        rows.append((sid, name, score, now, source_image))
    conn.commit()
    conn.close()
    # write CSV
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'attendance_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id','student_name','score','timestamp','source_image'])
        writer.writerows(rows)
    return csv_path
