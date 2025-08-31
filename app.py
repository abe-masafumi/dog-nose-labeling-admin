
from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import os
import json
import csv
from datetime import datetime
import shutil
from pathlib import Path
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dog-nose-labeling-secret-key'

@app.route('/export')
def export_screen():
    return render_template('export.html', active_page='export')

# 画像1枚分のデータセット（画像＋YOLOラベルtxt）zipのみをエクスポート
@app.route('/api/export_single/<int:image_id>')
def export_single_image_dataset(image_id):
    import zipfile
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT i.filename, i.filepath, l.main_label, l.bbox
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE i.id = ?
    ''', (image_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': '画像が見つかりません'}), 404
    filename, filepath, main_label, bbox = row
    # 一時ディレクトリ作成
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # 画像コピー
        img_out = os.path.join(tmpdir, filename)
        try:
            import shutil
            shutil.copy2(filepath, img_out)
        except Exception as e:
            return jsonify({'error': f'画像コピー失敗: {e}'}), 500
        # YOLOラベルtxt作成
        label_txt = os.path.join(tmpdir, os.path.splitext(filename)[0] + '.txt')
        yolo_line = ''
        if main_label == 'nose' and bbox:
            try:
                bbox_data = json.loads(bbox) if isinstance(bbox, str) else bbox
                if isinstance(bbox_data, dict):
                    if all(k in bbox_data for k in ('x', 'y', 'width', 'height')):
                        x = bbox_data.get('x')
                        y = bbox_data.get('y')
                        w = bbox_data.get('width')
                        h = bbox_data.get('height')
                    elif all(k in bbox_data for k in ('x_min', 'y_min', 'x_max', 'y_max')):
                        x = bbox_data['x_min']
                        y = bbox_data['y_min']
                        w = bbox_data['x_max'] - bbox_data['x_min']
                        h = bbox_data['y_max'] - bbox_data['y_min']
                    else:
                        x = y = w = h = None
                elif isinstance(bbox_data, list) and len(bbox_data) == 4:
                    x, y, w, h = bbox_data
                else:
                    x = y = w = h = None
                if None not in (x, y, w, h):
                    from PIL import Image
                    with Image.open(filepath) as im:
                        img_w, img_h = im.size
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            except Exception as e:
                print(f"YOLOラベル変換エラー: {filepath} -> {e}")
        with open(label_txt, 'w', encoding='utf-8') as f:
            f.write(yolo_line + '\n')
        # zip作成
        zip_path = os.path.join(tmpdir, f'single_dataset_{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(img_out, arcname=filename)
            zipf.write(label_txt, arcname=os.path.basename(label_txt))
        return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

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
            bbox TEXT,
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
    return render_template('index.html', active_page='label')

@app.route('/review')
def review():
    return render_template('review.html', active_page='review')

@app.route('/api/images')
def get_images():
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.bbox, l.is_completed
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
            'dataset_split': row[5],
            'bbox': row[6],
            'is_completed': row[7]
        })
    conn.close()
    return jsonify(images)

@app.route('/api/images/<int:image_id>')
def get_image(image_id):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.bbox
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
            'dataset_split': row[5],
            'bbox': row[6]
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
    bbox = data.get('bbox')
    if bbox is not None and not isinstance(bbox, str):
        bbox = json.dumps(bbox)
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    # is_completed=1を必ずセット
    # 既存レコードがあればUPDATE、なければINSERT
    cursor.execute('''
        INSERT INTO labels (image_path, main_label, sub_labels, dataset_split, bbox, is_completed, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(image_path) DO UPDATE SET
            main_label=excluded.main_label,
            sub_labels=excluded.sub_labels,
            dataset_split=excluded.dataset_split,
            bbox=excluded.bbox,
            is_completed=1,
            updated_at=CURRENT_TIMESTAMP
    ''', (image_path, main_label, sub_labels, dataset_split, bbox))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/export/dataset')
def export_dataset():
    """機械学習用フォルダ分割出力"""
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.filepath, l.main_label, l.dataset_split, l.bbox, l.is_completed
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_completed = 1 AND l.dataset_split IS NOT NULL
    ''')
    
    base_dir = 'data'
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    # 既存ディレクトリ削除
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return jsonify({'error': 'エクスポートするデータがありません'}), 400

    for row in rows:
        filepath, main_label, dataset_split, bbox, is_completed = row
        if not os.path.exists(filepath) or not dataset_split:
            continue
        # 画像コピー
        img_out_dir = os.path.join(images_dir, dataset_split)
        os.makedirs(img_out_dir, exist_ok=True)
        filename = os.path.basename(filepath)
        try:
            shutil.copy2(filepath, os.path.join(img_out_dir, filename))
        except Exception as e:
            print(f"ファイルコピーエラー: {filepath} -> {e}")
            continue
        # YOLOラベル出力
        label_out_dir = os.path.join(labels_dir, dataset_split)
        os.makedirs(label_out_dir, exist_ok=True)
        label_path = os.path.join(label_out_dir, os.path.splitext(filename)[0] + '.txt')
        yolo_lines = []
        # bboxがあればYOLO形式で出力
        if bbox:
            try:
                bbox_data = json.loads(bbox) if isinstance(bbox, str) else bbox
                # bboxは[x, y, w, h] or dict型想定
                if isinstance(bbox_data, dict):
                    if all(k in bbox_data for k in ('x', 'y', 'width', 'height')):
                        x = bbox_data.get('x')
                        y = bbox_data.get('y')
                        w = bbox_data.get('width')
                        h = bbox_data.get('height')
                    elif all(k in bbox_data for k in ('x_min', 'y_min', 'x_max', 'y_max')):
                        x = bbox_data['x_min']
                        y = bbox_data['y_min']
                        w = bbox_data['x_max'] - bbox_data['x_min']
                        h = bbox_data['y_max'] - bbox_data['y_min']
                    else:
                        x = y = w = h = None
                elif isinstance(bbox_data, list) and len(bbox_data) == 4:
                    x, y, w, h = bbox_data
                else:
                    x = y = w = h = None
                # YOLO形式: class x_center y_center width height (正規化済み)
                if None not in (x, y, w, h):
                    from PIL import Image
                    with Image.open(filepath) as im:
                        img_w, img_h = im.size
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            except Exception as e:
                print(f"YOLOラベル変換エラー: {filepath} -> {e}")
        # bboxがない場合は空ファイル
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
    
    conn.close()
    
    import zipfile
    zip_filename = f'dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        arcname = os.path.relpath(file_path, base_dir)
                        zipf.write(file_path, arcname)
        shutil.rmtree(base_dir)
        return send_file(zip_filename, as_attachment=True, download_name=zip_filename)
    except Exception as e:
        return jsonify({'error': f'ZIPファイル作成エラー: {str(e)}'}), 500

@app.route('/api/filter')
def filter_images():
    """フィルター条件に基づいて画像を検索"""
    main_label = request.args.get('main_label')
    sub_labels = request.args.getlist('sub_labels')
    dataset_split = request.args.get('dataset_split')
    
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    query = '''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE 1=1
    '''
    params = []
    
    if main_label:
        query += ' AND l.main_label = ?'
        params.append(main_label)
    
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
            'dataset_split': row[5]
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
        WHERE l.is_completed = 1
        ORDER BY i.id
    ''')
    
    completed_images = cursor.fetchall()
    
    if not completed_images:
        conn.close()
        return jsonify({'error': '作業済みの画像がありません'}), 400
    
    shuffled_images = list(completed_images)
    
    shuffled_images = list(completed_images)
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

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('images', filename))

if __name__ == '__main__':
    init_db()
    register_images()
    app.run(debug=True, host='0.0.0.0', port=8080)
