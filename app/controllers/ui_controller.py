from flask import render_template
import sqlite3
import json

def show_index():
    conn = sqlite3.connect("labels.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.id, i.filename, i.filepath, 
               l.main_label, l.sub_labels, l.dataset_split, l.bbox, l.is_completed, l.is_manual
        FROM images i
        LEFT JOIN labels l ON i.filepath = l.image_path
        ORDER BY i.id
    """)
    rows = cursor.fetchall()
    conn.close()

    images = []
    for row in rows:
        images.append({
            "id": row[0],
            "filename": row[1],
            "filepath": row[2],
            "main_label": row[3],
            "sub_labels": json.loads(row[4]) if row[4] else [],
            "dataset_split": row[5],
            "bbox": row[6],
            "is_completed": row[7],
            "is_manual": row[8]
        })

    return render_template("index.html", active_page="label", images=images)

def show_review():
    return render_template("review.html", active_page="review")

def show_export():
    return render_template("export.html", active_page="export")
