# YOLO推論用（ultralytics YOLOv8想定）
import os
import json
import sqlite3

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
