from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import os
import json
import csv
from datetime import datetime
import shutil
from pathlib import Path
import random
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dog-nose-labeling-secret-key'

def calculate_bbox_from_crop(original_path, cropped_path):
    """Calculate normalized bbox coordinates from original and cropped images"""
    try:
        original = Image.open(original_path)
        cropped = Image.open(cropped_path)
        
        orig_width, orig_height = original.size
        crop_width, crop_height = cropped.size
        
        x_center = 0.5
        y_center = 0.5
        
        width = crop_width / orig_width
        height = crop_height / orig_height
        
        return x_center, y_center, width, height
    except Exception as e:
        print(f"Error calculating bbox: {e}")
        return None, None, None, None

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
            is_reviewed INTEGER DEFAULT 0,
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
    
    try:
        cursor.execute('ALTER TABLE labels ADD COLUMN is_reviewed INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute('ALTER TABLE labels ADD COLUMN x_center REAL')
        cursor.execute('ALTER TABLE labels ADD COLUMN y_center REAL') 
        cursor.execute('ALTER TABLE labels ADD COLUMN width REAL')
        cursor.execute('ALTER TABLE labels ADD COLUMN height REAL')
    except sqlite3.OperationalError:
        pass
    
    cursor.execute('''
        UPDATE labels SET is_reviewed = 1 
        WHERE main_label IS NOT NULL AND is_reviewed = 0
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

@app.route('/review')
def review():
    return render_template('review.html')

@app.route('/api/images')
def get_images():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.is_reviewed,
               l.x_center, l.y_center, l.width, l.height
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        ORDER BY i.id
    ''')
    
    images = []
    for row in cursor.fetchall():
        sub_labels = json.loads(row[4]) if row[4] else []
        images.append({
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'main_label': row[3],
            'sub_labels': sub_labels,
            'dataset_split': row[5],
            'is_reviewed': row[6],
            'x_center': row[7],
            'y_center': row[8],
            'width': row[9],
            'height': row[10]
        })
    
    conn.close()
    return jsonify(images)

@app.route('/api/images/<int:image_id>')
def get_image(image_id):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.is_reviewed,
               l.x_center, l.y_center, l.width, l.height
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE i.id = ?
    ''', (image_id,))
    
    row = cursor.fetchone()
    if row:
        sub_labels = json.loads(row[4]) if row[4] else []
        image_data = {
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'main_label': row[3],
            'sub_labels': sub_labels,
            'dataset_split': row[5],
            'is_reviewed': row[6],
            'x_center': row[7],
            'y_center': row[8],
            'width': row[9],
            'height': row[10]
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
    is_reviewed = data.get('is_reviewed', 1)
    
    x_center = data.get('x_center')
    y_center = data.get('y_center') 
    width = data.get('width')
    height = data.get('height')
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO labels 
        (image_path, main_label, sub_labels, dataset_split, is_reviewed, 
         x_center, y_center, width, height, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ''', (image_path, main_label, sub_labels, dataset_split, is_reviewed,
          x_center, y_center, width, height))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/export/<format>')
def export_data(format):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.filename, i.filepath, l.main_label, l.sub_labels, l.dataset_split,
               l.x_center, l.y_center, l.width, l.height
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_reviewed = 1
    ''')
    
    data = []
    for row in cursor.fetchall():
        sub_labels = json.loads(row[3]) if row[3] else []
        data.append({
            'filename': row[0],
            'filepath': row[1],
            'main_label': row[2],
            'sub_labels': sub_labels,
            'dataset_split': row[4],
            'x_center': row[5],
            'y_center': row[6],
            'width': row[7],
            'height': row[8]
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
            writer.writerow(['filename', 'filepath', 'main_label', 'sub_labels', 'dataset_split', 'x_center', 'y_center', 'width', 'height'])
            for item in data:
                writer.writerow([
                    item['filename'],
                    item['filepath'],
                    item['main_label'],
                    ','.join(item['sub_labels']),
                    item['dataset_split'],
                    item['x_center'],
                    item['y_center'],
                    item['width'],
                    item['height']
                ])
        return send_file(filename, as_attachment=True)
    
    elif format == 'yolo':
        filename = f'yolo_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                if item['main_label'] == 'nose' and all(coord is not None for coord in [item['x_center'], item['y_center'], item['width'], item['height']]):
                    f.write(f"0 {item['x_center']} {item['y_center']} {item['width']} {item['height']}  # {item['filename']}\n")
        return send_file(filename, as_attachment=True)
    
    return jsonify({'error': '無効なフォーマットです'}), 400

@app.route('/api/export/dataset')
def export_dataset():
    """YOLO形式データセットエクスポート"""
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.filepath, l.main_label, l.dataset_split, 
               l.x_center, l.y_center, l.width, l.height
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_reviewed = 1 AND l.dataset_split IS NOT NULL
    ''')
    
    dataset_dir = 'dataset'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return jsonify({'error': 'エクスポートするデータがありません'}), 400
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)
    
    for row in rows:
        filepath, main_label, dataset_split, x_center, y_center, width, height = row
        
        if not os.path.exists(filepath):
            continue
            
        filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(filename)[0]
        
        image_output_dir = os.path.join(dataset_dir, 'images', dataset_split)
        try:
            shutil.copy2(filepath, os.path.join(image_output_dir, filename))
        except Exception as e:
            print(f"ファイルコピーエラー: {filepath} -> {e}")
            continue
        
        label_output_dir = os.path.join(dataset_dir, 'labels', dataset_split)
        label_filename = f"{filename_no_ext}.txt"
        
        with open(os.path.join(label_output_dir, label_filename), 'w') as f:
            if main_label == 'nose' and all(coord is not None for coord in [x_center, y_center, width, height]):
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
    
    dataset_yaml_content = f"""# YOLO dataset configuration
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['nose']
"""
    
    with open(os.path.join(dataset_dir, 'dataset.yaml'), 'w') as f:
        f.write(dataset_yaml_content)
    
    conn.close()
    
    import zipfile
    zip_filename = f'yolo_dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        arcname = os.path.relpath(file_path, dataset_dir)
                        zipf.write(file_path, arcname)
        
        shutil.rmtree(dataset_dir)
        
        return send_file(zip_filename, as_attachment=True, download_name=zip_filename)
    except Exception as e:
        return jsonify({'error': f'ZIPファイル作成エラー: {str(e)}'}), 500

@app.route('/api/filter')
def filter_images():
    """フィルター条件に基づいて画像を検索"""
    main_labels = request.args.getlist('main_label')
    sub_labels = request.args.getlist('sub_labels')
    dataset_split = request.args.get('dataset_split')
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    query = '''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.is_reviewed,
               l.x_center, l.y_center, l.width, l.height
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE 1=1
    '''
    params = []
    
    if main_labels:
        main_label_conditions = []
        for main_label in main_labels:
            if main_label == '':
                main_label_conditions.append('l.main_label IS NULL')
            else:
                main_label_conditions.append('l.main_label = ?')
                params.append(main_label)
        
        if main_label_conditions:
            query += ' AND (' + ' OR '.join(main_label_conditions) + ')'
    
    if dataset_split:
        query += ' AND l.dataset_split = ?'
        params.append(dataset_split)
    
    if sub_labels:
        for sub_label in sub_labels:
            query += ' AND l.sub_labels LIKE ?'
            params.append(f'%"{sub_label}"%')
    
    query += ' ORDER BY i.id'
    
    cursor.execute(query, params)
    
    results = []
    for row in cursor.fetchall():
        parsed_sub_labels = json.loads(row[4]) if row[4] else []
        results.append({
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'main_label': row[3],
            'sub_labels': parsed_sub_labels,
            'dataset_split': row[5],
            'is_reviewed': row[6],
            'x_center': row[7],
            'y_center': row[8],
            'width': row[9],
            'height': row[10]
        })
    
    conn.close()
    return jsonify(results)

@app.route('/api/auto-split', methods=['POST'])
def auto_split_dataset():
    """自動データセット分割（基本モード）"""
    data = request.json
    preview_only = data.get('preview_only', False)
    
    train_percent = data.get('train_percent', 80)
    val_percent = data.get('val_percent', 10) 
    test_percent = data.get('test_percent', 10)
    
    total = train_percent + val_percent + test_percent
    if abs(total - 100) > 0.01:
        return jsonify({'error': f'割合の合計が100%になりません: {total}%'}), 400
    
    if train_percent < 0 or val_percent < 0 or test_percent < 0:
        return jsonify({'error': '割合は0以上で入力してください'}), 400
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filepath, l.main_label, l.dataset_split
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_reviewed = 1
        ORDER BY i.id
    ''')
    
    labeled_images = cursor.fetchall()
    
    if not labeled_images:
        conn.close()
        return jsonify({'error': 'チェック済みの画像がありません'}), 400
    
    shuffled_images = list(labeled_images)
    random.shuffle(shuffled_images)
    
    total_count = len(shuffled_images)
    train_count = int(total_count * train_percent / 100)
    val_count = int(total_count * val_percent / 100)
    test_count = total_count - train_count - val_count
    
    train_images = shuffled_images[:train_count]
    val_images = shuffled_images[train_count:train_count + val_count]
    test_images = shuffled_images[train_count + val_count:]
    
    result = {
        'total_images': total_count,
        'train_count': len(train_images),
        'val_count': len(val_images), 
        'test_count': len(test_images),
        'train_percent_actual': round(len(train_images) / total_count * 100, 1),
        'val_percent_actual': round(len(val_images) / total_count * 100, 1),
        'test_percent_actual': round(len(test_images) / total_count * 100, 1)
    }
    
    if preview_only:
        conn.close()
        return jsonify(result)
    
    try:
        for image_id, filepath, main_label, current_split in train_images:
            cursor.execute('''
                UPDATE labels SET dataset_split = 'train', updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ?
            ''', (filepath,))
        
        for image_id, filepath, main_label, current_split in val_images:
            cursor.execute('''
                UPDATE labels SET dataset_split = 'val', updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ?
            ''', (filepath,))
        
        for image_id, filepath, main_label, current_split in test_images:
            cursor.execute('''
                UPDATE labels SET dataset_split = 'test', updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ?
            ''', (filepath,))
        
        conn.commit()
        result['success'] = True
        result['message'] = 'データセット分割が完了しました'
        
    except Exception as e:
        conn.rollback()
        result['error'] = f'データベース更新エラー: {str(e)}'
        return jsonify(result), 500
    finally:
        conn.close()
    
    return jsonify(result)

@app.route('/api/calculate-bbox', methods=['POST'])
def calculate_bbox():
    """Calculate bbox coordinates from original and cropped images"""
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'error': 'image_path required'}), 400
    
    filename = os.path.basename(image_path)
    cropped_path = os.path.join('cropped_image', filename)
    
    if not os.path.exists(cropped_path):
        return jsonify({'error': 'Cropped image not found'}), 404
    
    x_center, y_center, width, height = calculate_bbox_from_crop(image_path, cropped_path)
    
    if x_center is None:
        return jsonify({'error': 'Failed to calculate bbox'}), 500
    
    return jsonify({
        'x_center': x_center,
        'y_center': y_center, 
        'width': width,
        'height': height
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('images', filename))

if __name__ == '__main__':
    init_db()
    register_images()
    app.run(debug=True, host='0.0.0.0', port=8081)
