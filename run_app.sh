#!/usr/bin/env bash
set -e
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Starting Flask app on http://127.0.0.1:5000"
FLASK_APP=app.py FLASK_ENV=development flask run --host=127.0.0.1 --port=5000
