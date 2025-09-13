from flask import jsonify, request

def save_label():
    data = request.json
    return jsonify({"success": True, "data": data})

def export_single_image_dataset(image_id):
    return jsonify({"message": f"Export single image {image_id}"})

def export_dataset():
    return jsonify({"message": "Export dataset"})

def filter_images():
    return jsonify([])

def auto_split_dataset():
    return jsonify({"success": True})
