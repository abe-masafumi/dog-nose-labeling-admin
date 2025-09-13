import sys

# YOLO推論用（ultralytics YOLOv8想定）
def detect_nose_for_all_images(model_path='models/8_30_best.pt'):
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print('ultralytics, opencv-pythonが必要です。requirements.txtに追加してください。')
        return

    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    # is_manual=1以外の画像を抽出
    cursor.execute('''
        SELECT i.filepath, i.id FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_manual IS NULL
    ''')
    rows = cursor.fetchall()
    if not rows:
        print('推論対象画像がありません')
        conn.close()
        return

    model = YOLO(model_path)
    for filepath, image_id in rows:
        if not os.path.exists(filepath):
            print(f'Skip: {filepath} (not found)')
            continue
        img = cv2.imread(filepath)
        if img is None:
            print(f'Skip: {filepath} (cv2 load error)')
            continue
        results = model(img)
        if not results or not results[0].boxes:
            print(f'No detection: {filepath}')
            continue
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item() if hasattr(boxes.conf, 'argmax') else 0
        box = boxes[best_idx]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        bbox = {'x_min': x1, 'y_min': y1, 'x_max': x2, 'y_max': y2}
        # 既存レコードがあるか確認
        cursor.execute('SELECT COUNT(*) FROM labels WHERE image_path = ?', (filepath,))
        exists = cursor.fetchone()[0] > 0
        if exists:
            # bbox, main_label, is_manual, updated_atのみ更新（is_completedは自動で変更しない）
            cursor.execute('''
                UPDATE labels SET bbox = ?, main_label = ?, is_manual = 'auto', updated_at = CURRENT_TIMESTAMP
                WHERE image_path = ?
            ''', (json.dumps(bbox), 'nose', filepath))
        else:
            # 新規レコード挿入（is_completedはNULL/未設定で挿入）
            cursor.execute('''
                INSERT INTO labels (image_path, main_label, sub_labels, dataset_split, bbox, is_manual, updated_at)
                VALUES (?, ?, ?, ?, ?, 'auto', CURRENT_TIMESTAMP)
            ''', (filepath, 'nose', '[]', None, json.dumps(bbox)))
        print(f'Detected and saved: {filepath}')
    conn.commit()
    conn.close()
    print('鼻検出バッチ完了')
from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import os
import json
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
            is_manual BOOLEAN DEFAULT 0,
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
    # 自動データセット分割の設定保存用（履歴テーブル：実行ごとに新規行を追加）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auto_split_settings (
            id INTEGER PRIMARY KEY,
            train_percent REAL NOT NULL,
            val_percent REAL NOT NULL,
            test_percent REAL NOT NULL,
            targets TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
               l.main_label, l.sub_labels, l.dataset_split, l.bbox, l.is_completed, l.is_manual
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
            'is_completed': row[7],
            'is_manual': row[8] if row[8] is not None else None
        })
    conn.close()
    return jsonify(images)

@app.route('/api/images/<int:image_id>')
def get_image(image_id):
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.bbox, l.is_manual
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
            'bbox': row[6],
            'is_manual': row[7] if row[7] is not None else None
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
    # 鼻の長さサブラベルが1つも選択されていない場合はis_completed=0で保存
    try:
        sub_labels_list = json.loads(sub_labels)
    except Exception:
        sub_labels_list = []
    nose_length_labels = {'nose_long', 'nose_medium', 'nose_short'}
    has_nose_length = any(lbl in sub_labels_list for lbl in nose_length_labels)
    is_completed = 1
    if main_label == 'nose' and not has_nose_length:
        is_completed = 0

    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO labels (image_path, main_label, sub_labels, dataset_split, bbox, is_completed, is_manual, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'manual', CURRENT_TIMESTAMP)
        ON CONFLICT(image_path) DO UPDATE SET
            main_label=excluded.main_label,
            sub_labels=excluded.sub_labels,
            dataset_split=excluded.dataset_split,
            bbox=excluded.bbox,
            is_completed=excluded.is_completed,
            is_manual='manual',
            updated_at=CURRENT_TIMESTAMP
    ''', (image_path, main_label, sub_labels, dataset_split, bbox, is_completed))
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
        WHERE l.is_completed = 1
    '''
    params = []
    
    if main_label:
        query += ' AND l.main_label = ?'
        params.append(main_label)
    
    if dataset_split:
        query += ' AND l.dataset_split = ?'
        params.append(dataset_split)
    
    if sub_labels:
        or_clauses = []
        for sub_label in sub_labels:
            or_clauses.append('l.sub_labels LIKE ?')
            params.append(f'%"{sub_label}"%')
        if or_clauses:
            query += ' AND (' + ' OR '.join(or_clauses) + ')'
    
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
    """自動データセット分割（詳細ターゲット対応）"""
    data = request.json
    preview_only = data.get('preview_only', False)

    train_percent = float(data.get('train_percent', 80))
    val_percent = float(data.get('val_percent', 10))
    test_percent = float(data.get('test_percent', 10))
    targets = data.get('targets', {}) or {}

    total = train_percent + val_percent + test_percent
    if abs(total - 100) > 0.01:
        return jsonify({'error': f'割合の合計が100%になりません: {total}%'}), 400

    if train_percent < 0 or val_percent < 0 or test_percent < 0:
        return jsonify({'error': '割合は0以上で入力してください'}), 400

    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()

    # 設定保存は「実行時（preview_only=False）」にのみ行う

    # クラス設定済み（main_label が null/空でない）を対象に取得
    cursor.execute('''
        SELECT i.id, i.filepath, l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.main_label IS NOT NULL AND l.main_label != ''
        ORDER BY i.id
    ''')

    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return jsonify({'error': '作業済みの画像がありません'}), 400

    # 画像プール準備
    pool = []
    for image_id, filepath, main_label, sub_labels_json, current_split in rows:
        try:
            sub_labels = json.loads(sub_labels_json) if sub_labels_json else []
        except Exception:
            sub_labels = []
        pool.append({
            'id': image_id,
            'filepath': filepath,
            'main_label': main_label,
            'sub_labels': sub_labels,
        })

    total_count = len(pool)
    # 元のシンプル分割（ターゲット未指定時）に後方互換
    if not any(isinstance(v, dict) and v for v in targets.values()):
        rng = list(pool)
        random.shuffle(rng)
        train_count = int(total_count * train_percent / 100)
        val_count = int(total_count * val_percent / 100)
        test_count = total_count - train_count - val_count

        train_images = rng[:train_count]
        val_images = rng[train_count:train_count + val_count]
        test_images = rng[train_count + val_count:]

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
            for item in train_images:
                cursor.execute("UPDATE labels SET dataset_split = 'train', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
            for item in val_images:
                cursor.execute("UPDATE labels SET dataset_split = 'val', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
            for item in test_images:
                cursor.execute("UPDATE labels SET dataset_split = 'test', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
            # 実行成功時のみ設定を新規保存（履歴追加）
            cursor.execute('''
                INSERT INTO auto_split_settings (train_percent, val_percent, test_percent, targets, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (train_percent, val_percent, test_percent, json.dumps(targets)))
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

    # 詳細ターゲットあり：貪欲割当
    splits = ['train', 'val', 'test']
    split_perc = {
        'train': train_percent,
        'val': val_percent,
        'test': test_percent,
    }
    split_targets_counts = {s: int(total_count * split_perc[s] / 100) for s in splits}

    # 属性マップと判定関数
    attribute_order = ['orientation', 'clarity', 'color', 'size', 'fur', 'nose_length', 'main_label']

    def matches(item, attr, val):
        if attr == 'main_label':
            ml = item.get('main_label')
            if val == 'none':
                return (ml is None) or (ml == '') or (ml == 'none')
            return (val == ml)
        # sub_labels に含まれるかで判定
        return val in (item.get('sub_labels') or [])

    # 割当結果
    assignments = {s: [] for s in splits}
    # 現在のカバレッジ集計
    coverage = {s: {attr: {} for attr in attribute_order} for s in splits}

    # 乱択順
    unassigned = list(pool)
    random.shuffle(unassigned)

    # 各 split でターゲットを満たすように追加
    for s in splits:
        target_total = split_targets_counts[s]
        # まずは指定された属性ターゲットを順に満たす
        for attr in attribute_order:
            attr_targets = (targets.get(attr) or {})
            # 値ごとのパーセンテージ取得
            for val, conf in attr_targets.items():
                try:
                    percent = float((conf or {}).get(s, 0) or 0)
                except Exception:
                    percent = 0
                if percent <= 0:
                    continue
                required = int(target_total * percent / 100)
                # すでに割当済みの該当数
                current = coverage[s][attr].get(val, 0)
                need = max(0, required - current)
                if need <= 0:
                    continue
                # 未割当から条件に合うものを取得
                picked = []
                remain = []
                for item in unassigned:
                    if len(picked) < need and matches(item, attr, val):
                        picked.append(item)
                    else:
                        remain.append(item)
                unassigned = remain
                assignments[s].extend(picked)
                # カバレッジは最終集計時に assignments から一括で更新する（途中での二重加算を避ける）
                # split の総数を超えないように調整
                if len(assignments[s]) >= target_total:
                    break
            if len(assignments[s]) >= target_total:
                break
        # 余りを埋める（ターゲット未指定 or 充足後の残り）
        remain_needed = max(0, target_total - len(assignments[s]))
        if remain_needed > 0 and unassigned:
            take = unassigned[:remain_needed]
            assignments[s].extend(take)
            unassigned = unassigned[remain_needed:]
        # カバレッジを assignments から一括更新（main_label と、targets に指定があるサブラベルのみ）
        for item in assignments[s]:
            ml = item.get('main_label')
            coverage[s]['main_label'][ml] = coverage[s]['main_label'].get(ml, 0) + 1
            for attr in ['orientation', 'clarity', 'color', 'size', 'fur', 'nose_length']:
                attr_values = targets.get(attr)
                if not attr_values:
                    continue
                for lbl in (item.get('sub_labels') or []):
                    if lbl in attr_values:
                        coverage[s][attr][lbl] = coverage[s][attr].get(lbl, 0) + 1

    train_images = assignments['train']
    val_images = assignments['val']
    test_images = assignments['test']

    result = {
        'total_images': total_count,
        'train_count': len(train_images),
        'val_count': len(val_images),
        'test_count': len(test_images),
        'train_percent_actual': round(len(train_images) / total_count * 100, 1) if total_count else 0,
        'val_percent_actual': round(len(val_images) / total_count * 100, 1) if total_count else 0,
        'test_percent_actual': round(len(test_images) / total_count * 100, 1) if total_count else 0,
        'details': {
            s: {
                attr: {
                    'actual_counts': coverage[s][attr],
                    'required_counts': {
                        val: int(split_targets_counts[s] * float(((targets.get(attr) or {}).get(val) or {}).get(s, 0) or 0) / 100)
                        for val in (targets.get(attr) or {}).keys()
                    }
                } for attr in attribute_order if targets.get(attr)
            } for s in splits
        }
    }

    if preview_only:
        conn.close()
        return jsonify(result)

    # 実更新
    try:
        for item in train_images:
            cursor.execute("UPDATE labels SET dataset_split = 'train', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
        for item in val_images:
            cursor.execute("UPDATE labels SET dataset_split = 'val', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
        for item in test_images:
            cursor.execute("UPDATE labels SET dataset_split = 'test', updated_at = CURRENT_TIMESTAMP WHERE image_path = ?", (item['filepath'],))
        # 実行成功時のみ設定を新規保存（履歴追加）
        cursor.execute('''
            INSERT INTO auto_split_settings (train_percent, val_percent, test_percent, targets, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (train_percent, val_percent, test_percent, json.dumps(targets)))
        conn.commit()
        result['success'] = True
        result['message'] = 'データセット分割が完了しました（詳細ターゲット反映）'
    except Exception as e:
        conn.rollback()
        result['error'] = f'データベース更新エラー: {str(e)}'
        return jsonify(result), 500
    finally:
        conn.close()

    return jsonify(result)


@app.route('/api/auto-split/settings', methods=['GET'])
def get_auto_split_settings():
    """保存済みの自動分割設定を返す（無ければデフォルト）"""
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    cursor.execute('SELECT train_percent, val_percent, test_percent, targets, updated_at FROM auto_split_settings ORDER BY updated_at DESC, id DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({
            'train_percent': 80.0,
            'val_percent': 10.0,
            'test_percent': 10.0,
            'targets': {},
            'updated_at': None
        })
    train_percent, val_percent, test_percent, targets_text, updated_at = row
    try:
        targets = json.loads(targets_text) if targets_text else {}
    except Exception:
        targets = {}
    return jsonify({
        'train_percent': train_percent,
        'val_percent': val_percent,
        'test_percent': test_percent,
        'targets': targets,
        'updated_at': updated_at
    })


@app.route('/api/auto-split/settings', methods=['POST'])
def save_auto_split_settings():
    """自動分割設定を保存（部分更新可・上書き保存）
    注意: 現仕様では UI は最新の実行結果のみ表示するため GET のみ使用。
    本POSTは互換用に残置（必要なら手動で設定を差し替え可能）。
    """
    data = request.json or {}
    conn = sqlite3.connect('labels.db')
    cursor = conn.cursor()
    # 現在値を取得
    cursor.execute('SELECT train_percent, val_percent, test_percent, targets FROM auto_split_settings WHERE id = 1')
    row = cursor.fetchone()
    if row:
        cur_train, cur_val, cur_test, cur_targets = row
        try:
            cur_targets = json.loads(cur_targets) if cur_targets else {}
        except Exception:
            cur_targets = {}
    else:
        cur_train, cur_val, cur_test, cur_targets = 80.0, 10.0, 10.0, {}

    # 入力をマージ（与えられたもののみ上書き）
    train_percent = float(data.get('train_percent', cur_train)) if data.get('train_percent') is not None else cur_train
    val_percent = float(data.get('val_percent', cur_val)) if data.get('val_percent') is not None else cur_val
    test_percent = float(data.get('test_percent', cur_test)) if data.get('test_percent') is not None else cur_test
    new_targets = data.get('targets', None)
    if new_targets is None:
        merged_targets = cur_targets
    else:
        # 完全置換（部分マージを望む場合はここでディープマージへ変更可）
        merged_targets = new_targets

    try:
        cursor.execute('''
            INSERT INTO auto_split_settings (id, train_percent, val_percent, test_percent, targets, updated_at)
            VALUES (1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                train_percent=excluded.train_percent,
                val_percent=excluded.val_percent,
                test_percent=excluded.test_percent,
                targets=excluded.targets,
                updated_at=CURRENT_TIMESTAMP
        ''', (train_percent, val_percent, test_percent, json.dumps(merged_targets)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'error': f'設定保存エラー: {str(e)}'}), 500
    conn.close()
    return jsonify({'success': True})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('images', filename))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'detect_nose':
        detect_nose_for_all_images()
    else:
        init_db()
        register_images()
        app.run(debug=True, host='0.0.0.0', port=8080)
