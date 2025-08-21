from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import os
import json
import csv
from datetime import datetime
import shutil
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dog-nose-labeling-secret-key'

def init_db():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE NOT NULL,
            main_label TEXT,
            sub_labels TEXT,
            dataset_split TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def register_images():
    images_dir = 'images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        return
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(images_dir, filename)
            file_size = os.path.getsize(filepath)
            
            cursor.execute('''
                INSERT OR IGNORE INTO images (filename, filepath, file_size)
                VALUES (?, ?, ?)
            ''', (filename, filepath, file_size))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/images')
def get_images():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        ORDER BY i.id
    ''')
    
    images = []
    for row in cursor.fetchall():
        images.append({
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'main_label': row[3],
            'sub_labels': row[4],
            'dataset_split': row[5]
        })
    
    conn.close()
    return jsonify(images)

@app.route('/api/images/<int:image_id>')
def get_image(image_id):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE i.id = ?
    ''', (image_id,))
    
    row = cursor.fetchone()
    if row:
        image_data = {
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'main_label': row[3],
            'sub_labels': row[4],
            'dataset_split': row[5]
        }
        conn.close()
        return jsonify(image_data)
    
    conn.close()
    return jsonify({'error': '画像が見つかりません'}), 404

@app.route('/api/labels', methods=['POST'])
def save_label():
    data = request.json
    image_path = data.get('image_path')
    main_label = data.get('main_label')
    sub_labels = json.dumps(data.get('sub_labels', []))
    dataset_split = data.get('dataset_split')
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO labels 
        (image_path, main_label, sub_labels, dataset_split, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (image_path, main_label, sub_labels, dataset_split))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/export/<format>')
def export_data(format):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.filename, i.filepath, l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE l.main_label IS NOT NULL
    ''')
    
    data = []
    for row in cursor.fetchall():
        sub_labels = json.loads(row[3]) if row[3] else []
        data.append({
            'filename': row[0],
            'filepath': row[1],
            'main_label': row[2],
            'sub_labels': sub_labels,
            'dataset_split': row[4]
        })
    
    conn.close()
    
    if format == 'json':
        filename = f'labels_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return send_file(filename, as_attachment=True)
    
    elif format == 'csv':
        filename = f'labels_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'filepath', 'main_label', 'sub_labels', 'dataset_split'])
            for item in data:
                writer.writerow([
                    item['filename'],
                    item['filepath'],
                    item['main_label'],
                    ','.join(item['sub_labels']),
                    item['dataset_split']
                ])
        return send_file(filename, as_attachment=True)
    
    return jsonify({'error': '無効なフォーマットです'}), 400

@app.route('/api/export/dataset')
def export_dataset():
    """機械学習用フォルダ分割出力"""
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.filepath, l.main_label, l.dataset_split
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.main_label IS NOT NULL AND l.dataset_split IS NOT NULL
    ''')
    
    dataset_dir = 'dataset'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    for row in cursor.fetchall():
        filepath, main_label, dataset_split = row
        
        output_dir = os.path.join(dataset_dir, dataset_split, main_label)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(filepath)
        shutil.copy2(filepath, os.path.join(output_dir, filename))
    
    conn.close()
    
    import zipfile
    zip_filename = f'dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, arcname)
    
    return send_file(zip_filename, as_attachment=True)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('images', filename))

if __name__ == '__main__':
    init_db()
    register_images()
    app.run(debug=True, host='0.0.0.0', port=8080)
