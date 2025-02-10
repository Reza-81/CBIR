from flask import Flask, render_template, request, jsonify, send_from_directory, g
import os
import cv2
import numpy as np
import sqlite3
from werkzeug.utils import secure_filename
from similarity_system import SimilaritySystem
from database import DatabaseHandler
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'your-secret-key-here'

# Initialize feature extractor
feature_extractor = SimilaritySystem()

# Database configuration
DATABASE = 'cbir.db'

# Add this global variable at the top of app.py
feedback_store = {}

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

def get_db_handler():
    if 'db_handler' not in g:
        g.db_handler = DatabaseHandler(get_db())
    return g.db_handler

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

app.teardown_appcontext(close_db)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Generate a unique filename
        unique_id = uuid.uuid4().hex
        original_filename = secure_filename(file.filename)
        filename = f"{unique_id}_{original_filename}"  # Unique filename
        filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(filepath)
        return jsonify({
            "filename": filename,  # Return the unique filename
            "original_filename": original_filename  # Optionally return the original filename
        })
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/search', methods=['POST'])
def perform_search():
    data = request.json
    filename = data.get('filename')  # This should be the unique filename
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Use the unique filename
    query_image = cv2.imread(filepath)
    if query_image is None:
        return jsonify({"error": "Failed to load image"}), 400

    try:
        db_handler = get_db_handler()
        results = db_handler.search(query_image, feature_extractor, k=10)
        return jsonify({"results": format_results(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/refine_search', methods=['POST'])
def refine_search():
    data = request.json
    filename = data.get('filename')  # This should be the unique filename
    relevant_paths = data.get('relevant', [])
    non_relevant_paths = data.get('non_relevant', [])
    # Rocchio algorithm parameters
    alpha = 0.6
    beta = 0.8
    gamma = 0.8

    try:
        # Initialize feedback entry for the current query image
        if filename not in feedback_store:
            feedback_store[filename] = {
                'relevant': [],
                'non_relevant': [],
                'query_features': None
            }
        current_feedback = feedback_store[filename]

        db_handler = get_db_handler()

        # Process relevant images
        for path in relevant_paths:
            features = db_handler.get_image_features(path)
            current_feedback['relevant'].append(features)

        # Process non-relevant images
        for path in non_relevant_paths:
            features = db_handler.get_image_features(path)
            current_feedback['non_relevant'].append(features)

        # Load and process query image if not already processed
        if current_feedback['query_features'] is None:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Use the unique filename
            query_image = cv2.imread(filepath)
            original_features = feature_extractor.extract_features(query_image)
            original_features = original_features / np.linalg.norm(original_features)
            current_feedback['query_features'] = original_features
        else:
            original_features = current_feedback['query_features']

        # Compute average relevant features
        avg_relevant = np.zeros_like(original_features)
        if current_feedback['relevant']:
            relevant_features = np.array(current_feedback['relevant'], dtype=np.float32)
            relevant_norms = np.linalg.norm(relevant_features, axis=1, keepdims=True)
            normalized_relevant = relevant_features / np.clip(relevant_norms, 1e-8, None)
            avg_relevant = np.mean(normalized_relevant, axis=0)
            avg_relevant /= np.linalg.norm(avg_relevant) if np.linalg.norm(avg_relevant) > 0 else 1

        # Compute average non-relevant features
        avg_non_relevant = np.zeros_like(original_features)
        if current_feedback['non_relevant']:
            non_rel_features = np.array(current_feedback['non_relevant'], dtype=np.float32)
            non_rel_norms = np.linalg.norm(non_rel_features, axis=1, keepdims=True)
            normalized_non_rel = non_rel_features / np.clip(non_rel_norms, 1e-8, None)
            avg_non_relevant = np.mean(normalized_non_rel, axis=0)
            avg_non_relevant /= np.linalg.norm(avg_non_relevant) if np.linalg.norm(avg_non_relevant) > 0 else 1

        # Apply Rocchio's formula
        refined_query = alpha * original_features + beta * avg_relevant - gamma * avg_non_relevant
        refined_query /= np.linalg.norm(refined_query)

        # Perform search with refined features
        distances, indices = db_handler.index.search(np.array([refined_query]).astype('float32'), k=10)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            path = db_handler.get_image_path(int(idx) + 1)
            if path:
                normalized_score = (score + 1) / 2
                results.append((path, float(normalized_score)))

        return jsonify({"results": format_results(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def format_results(results):
    return [{
        "path": path,
        "url": f"/get_image/{path}",
        "score": score
    } for path, score in results]

@app.route('/get_image/<path:image_path>')
def get_image(image_path):
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)