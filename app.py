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
import hashlib

app = Flask(__name__, static_folder='app/static', static_url_path='/static')
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
    # 互換マイグレーション: labels に is_completed 列が無ければ追加
    try:
        cursor.execute("PRAGMA table_info(labels)")
        cols = [r[1] for r in cursor.fetchall()]
        if 'is_completed' not in cols:
            cursor.execute("ALTER TABLE labels ADD COLUMN is_completed INTEGER DEFAULT 0")
            conn.commit()
    except Exception as e:
        # 追加に失敗しても続行（既存環境では既に存在する想定）
        pass
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
    # 入力を堅牢に受け取る（壊れたJSONでの500回避）
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        return jsonify({'error': '無効なJSONリクエストです'}), 400
    preview_only = data.get('preview_only', False)
    prefer_target_distribution = bool(data.get('prefer_target_distribution', False))

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

    # 対象データ: is_completed = 1 のみ（鼻なし含む）
    cursor.execute('''
        SELECT i.id, i.filepath, l.main_label, l.sub_labels, l.dataset_split
        FROM images i
        JOIN labels l ON i.filepath = l.image_path
        WHERE l.is_completed = 1
        ORDER BY i.id
    ''')

    rows = cursor.fetchall()
    if not rows:
        conn.close()
        return jsonify({'error': '分割対象（is_completed=1）の画像がありません'}), 400

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

    # オプション: ターゲットが main_label のみで、各splitが nose=100%（noneほか0%）の場合は、
    # プールを鼻あり（main_label == 'nose'相当）のみへ縮小して「未使用を許容」する
    def targets_require_nose_only(tg):
        if not tg:
            return False
        cls = (tg.get('main_label') or {})
        # 値は 'nose' と 'none' を想定（その他キーがあれば不一致）
        allowed_keys = set(['nose', 'none'])
        if any(k not in allowed_keys for k in cls.keys()):
            return False
        # 各splitについて nose=100 かつ none が未指定または0 とみなせる場合のみ True
        def pct_of(val, split):
            try:
                return float(((cls.get(val) or {}).get(split, 0)) or 0)
            except Exception:
                return 0.0
        for split in ['train','val','test']:
            if abs(pct_of('nose', split) - 100.0) > 1e-6:
                return False
            if pct_of('none', split) > 0:
                return False
        return True

    # prefer が有効で、かつ main_label で各splitが nose=100%（none=0%/未指定）なら鼻ありのみ使用
    if prefer_target_distribution and targets_require_nose_only(targets):
        filtered = []
        for p in pool:
            ml = p.get('main_label')
            if ml is None or ml == '' or ml == 'none':
                continue
            if ml == 'nose':
                filtered.append(p)
        pool = filtered

    total_count = len(pool)
    # 決定的シード（プレビューと実行で同一結果に）
    def build_seed():
        fp_concat = '\n'.join(sorted([p['filepath'] for p in pool]))
        targets_text = json.dumps(targets or {}, ensure_ascii=False, sort_keys=True)
        s = f"{train_percent}-{val_percent}-{test_percent}\n" + fp_concat + "\n" + targets_text
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**32)

    rng = random.Random(build_seed())

    has_targets = any(isinstance(v, dict) and v for v in targets.values())
    # 元のシンプル分割（ターゲット未指定時）に後方互換
    if not has_targets:
        lst = list(pool)
        rng.shuffle(lst)
        # ハミルトン法で希望枚数を算出
        exact = {
            'train': total_count * train_percent / 100.0,
            'val': total_count * val_percent / 100.0,
            'test': total_count * test_percent / 100.0,
        }
        base = {k: int(v) for k, v in exact.items()}
        assigned = sum(base.values())
        rema = sorted([(exact[s] - base[s], s) for s in ['train','val','test']], reverse=True)
        iidx = 0
        while assigned < total_count:
            _, sname = rema[iidx % len(rema)]
            base[sname] += 1
            assigned += 1
            iidx += 1
        train_count, val_count, test_count = base['train'], base['val'], base['test']

        train_images = lst[:train_count]
        val_images = lst[train_count:train_count + val_count]
        test_images = lst[train_count + val_count:]

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
            # 既存の分割結果をクリア（対象: is_completed=1）
            cursor.execute("UPDATE labels SET dataset_split = NULL, updated_at = CURRENT_TIMESTAMP WHERE is_completed = 1")
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
    # prefer の有効化条件: 明示ONのみ（UIはデフォルトOFF）
    prefer_effective = prefer_target_distribution
    # ハミルトン法で split 希望枚数を算出（合計=total_count）
    exact = {s: total_count * split_perc[s] / 100.0 for s in splits}
    desired_counts = {s: int(exact[s]) for s in splits}
    assigned = sum(desired_counts.values())
    rema = sorted([(exact[s] - desired_counts[s], s) for s in splits], reverse=True)
    iidx = 0
    while assigned < total_count:
        _, sname = rema[iidx % len(rema)]
        desired_counts[sname] += 1
        assigned += 1
        iidx += 1

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

    def is_none_item(it):
        ml = it.get('main_label')
        return (ml is None) or (ml == '') or (ml == 'none')

    # 割当結果
    assignments = {s: [] for s in splits}
    # 現在のカバレッジ集計
    coverage = {s: {attr: {} for attr in attribute_order} for s in splits}

    # 乱択順
    unassigned = list(pool)
    rng.shuffle(unassigned)

    # ターゲット重視時のフェアネス: main_label 'none'（鼻なし）の配分をavailable枚数で按分
    fairness_required = {s: {} for s in splits}
    if prefer_effective:
        try:
            # 可用な鼻なし枚数
            avail_none = sum(1 for p in pool if (p.get('main_label') in (None, '', 'none')))
            cls_targets = (targets.get('main_label') or {})
            # 各splitの重み（希望量）: desired_counts[s] * percent_none
            weights = {}
            sum_w = 0.0
            for s in splits:
                pct = float(((cls_targets.get('none') or {}).get(s, 0)) or 0)
                w = desired_counts[s] * pct / 100.0
                weights[s] = w
                sum_w += w
            if avail_none > 0 and sum_w > 0:
                # ハミルトン按分
                exact = {s: avail_none * (weights[s] / sum_w) for s in splits}
                base = {s: int(exact[s]) for s in splits}
                assigned = sum(base.values())
                rema = sorted([(exact[s] - base[s], s) for s in splits], reverse=True)
                iidx = 0
                while assigned < avail_none:
                    _, sname = rema[iidx % len(rema)]
                    base[sname] += 1
                    assigned += 1
                    iidx += 1
                for s in splits:
                    if base[s] > 0:
                        fairness_required[s]['main_label:none'] = base[s]
        except Exception:
            pass

    # noneの要求上限（prefer時のみ有効）
    required_none_limit = {s: int(fairness_required.get(s, {}).get('main_label:none', 0) or 0) for s in splits}

    

    # 事前割当（prefer時の希少クラス: main_label 'none'）
    if prefer_effective:
        # 公平按分で要求された 'none' を先に確保
        none_indices = [idx for idx, it in enumerate(unassigned) if is_none_item(it)]
        take_order = none_indices  # 既にrng.shuffle済みの順を尊重
        used_indices = set()
        for s in ['val', 'test', 'train']:
            need_none = required_none_limit.get(s, 0)
            if need_none <= 0:
                continue
            target_total = desired_counts[s]
            # splitの目標枚数を超えないように確保
            can_take = max(0, target_total - len(assignments[s]))
            need = min(need_none, can_take)
            for idx in take_order:
                if need <= 0:
                    break
                if idx in used_indices:
                    continue
                item = unassigned[idx]
                # 念のため None 判定
                if not is_none_item(item):
                    continue
                assignments[s].append(item)
                used_indices.add(idx)
                need -= 1
        if used_indices:
            # 未割当に残す要素のみ再構築
            unassigned = [it for i, it in enumerate(unassigned) if i not in used_indices]

    # 各 split でターゲットを満たすように追加
    for s in splits:
        target_total = desired_counts[s]
        # まずは指定された属性ターゲットを順に満たす
        for attr in attribute_order:
            attr_targets = (targets.get(attr) or {})
            # 値ごとの割当順序: prefer有効時かつ main_label の場合は 'none' を先に処理して枠を確保
            if prefer_effective and attr == 'main_label' and attr_targets:
                vals_order = sorted(list(attr_targets.keys()), key=lambda v: 0 if v == 'none' else 1)
            else:
                vals_order = list(attr_targets.keys())
            # 値ごとのパーセンテージ取得
            for val in vals_order:
                conf = attr_targets.get(val) or {}
                try:
                    percent = float((conf or {}).get(s, 0) or 0)
                except Exception:
                    percent = 0
                if percent <= 0:
                    continue
                # フェアネス上書き: main_label:none は按分済みの目標を使う
                if prefer_effective and attr == 'main_label' and val == 'none' and fairness_required[s].get('main_label:none') is not None:
                    required = int(fairness_required[s].get('main_label:none') or 0)
                else:
                    required = int(target_total * percent / 100)
                # すでに割当済みの該当数（coverageではなく、現assignmentsから算出して重複加算を防止）
                current = 0
                for _it in assignments[s]:
                    if matches(_it, attr, val):
                        current += 1
                need = max(0, required - current)
                if need <= 0:
                    continue
                # 未割当から条件に合うものを取得
                picked = []
                remain = []
                taken_indices = set()
                taken_none = 0
                for idx, item in enumerate(unassigned):
                    # split の目標枚数を超えないように保護
                    if len(assignments[s]) + len(picked) >= target_total:
                        remain.append(item)
                        continue
                    # prefer時は none の上限を超えないよう制御
                    if prefer_effective and attr != 'main_label' and is_none_item(item):
                        # すでに割当済みのnone数
                        assigned_none = sum(1 for _it in assignments[s] if is_none_item(_it)) + taken_none
                        if assigned_none >= required_none_limit.get(s, 0):
                            remain.append(item)
                            continue
                    if len(picked) < need and matches(item, attr, val):
                        picked.append(item)
                        if prefer_effective and attr != 'main_label' and is_none_item(item):
                            taken_none += 1
                        taken_indices.add(idx)
                    else:
                        remain.append(item)
                unassigned = [it for i, it in enumerate(unassigned) if i not in taken_indices]
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
            # prefer の有無にかかわらず、split の目標枚数までは未割当から充足する
            taken_indices = set()
            take = []
            taken_none = 0
            for idx, item in enumerate(unassigned):
                if len(take) >= remain_needed:
                    break
                if prefer_effective and is_none_item(item):
                    assigned_none = sum(1 for _it in assignments[s] if is_none_item(_it)) + taken_none
                    if assigned_none >= required_none_limit.get(s, 0):
                        continue
                    taken_none += 1
                take.append(item)
                taken_indices.add(idx)
            assignments[s].extend(take)
            if taken_indices:
                unassigned = [it for i, it in enumerate(unassigned) if i not in taken_indices]

    # 残余があれば少ない split から順に追加
    if unassigned:
        if prefer_effective:
            # ターゲット優先: 余剰は配分せず未使用（比率は近似のまま）
            unassigned = []
        else:
            order = sorted(splits, key=lambda x: len(assignments[x]))
            for item in unassigned:
                order = sorted(splits, key=lambda x: len(assignments[x]))
                assignments[order[0]].append(item)
            unassigned = []

    # 最終リバランス: prefer_target_distribution がオフのときだけ実施
    if not prefer_effective:
        def move_one(src, dst):
            if assignments[src]:
                assignments[dst].append(assignments[src].pop())
                return True
            return False

        # test を満たす
        while len(assignments['test']) < desired_counts['test']:
            moved = False
            for src in ['train', 'val']:
                if len(assignments[src]) > desired_counts[src]:
                    if move_one(src, 'test'):
                        moved = True
                        break
            if not moved:
                break
        # val を満たす
        while len(assignments['val']) < desired_counts['val']:
            moved = False
            for src in ['train', 'test']:
                if len(assignments[src]) > desired_counts[src]:
                    if move_one(src, 'val'):
                        moved = True
                        break
            if not moved:
                break

    # カバレッジを assignments から計算（main_label + 指定されたラベルのみ）
    for s in splits:
        for item in assignments[s]:
            # main_label は None/空文字を 'none' に正規化（JSONキーの安定化とUI整合のため）
            ml_raw = item.get('main_label')
            ml = ml_raw if (ml_raw is not None and ml_raw != '') else 'none'
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
                    'required_counts': (
                        {
                            **{
                                v: (
                                    # prefer時の main_label:none は公平按分を表示
                                    int(fairness_required[s].get('main_label:none', 0))
                                    if (prefer_effective and attr == 'main_label' and v == 'none' and fairness_required[s].get('main_label:none') is not None)
                                    else int(
                                        desired_counts[s]
                                        * float(((targets.get(attr) or {}).get(v) or {}).get(s, 0) or 0)
                                        / 100
                                    )
                                )
                                for v in (targets.get(attr) or {}).keys()
                            }
                        }
                    )
                }
                for attr in attribute_order if targets.get(attr)
            }
            for s in splits
        }
    }

    if preview_only:
        conn.close()
        return jsonify(result)

    # 実更新
    try:
        # 既存の分割結果をクリア（対象: is_completed=1）
        cursor.execute("UPDATE labels SET dataset_split = NULL, updated_at = CURRENT_TIMESTAMP WHERE is_completed = 1")
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
