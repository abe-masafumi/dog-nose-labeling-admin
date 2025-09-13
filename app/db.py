import sqlite3
import os

def init_db():
    conn = sqlite3.connect("labels.db")
    cursor = conn.cursor()

    # labels テーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE NOT NULL,
            main_label TEXT,
            sub_labels TEXT,
            dataset_split TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_completed INTEGER DEFAULT 0,
            bbox TEXT,
            is_manual BOOLEAN DEFAULT 0
        )
    """)

    # images テーブル
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def register_images():
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        return
    
    conn = sqlite3.connect("labels.db")
    cursor = conn.cursor()
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # PNGも追加
            filepath = os.path.join(images_dir, filename)
            file_size = os.path.getsize(filepath)
            
            cursor.execute('''
                INSERT OR IGNORE INTO images (filename, filepath, file_size)
                VALUES (?, ?, ?)
            ''', (filename, filepath, file_size))
    
    conn.commit()
    conn.close()
