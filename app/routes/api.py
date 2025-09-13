from flask import Blueprint
from app.controllers.api_controller import (
    save_label,
    export_single_image_dataset,
    export_dataset,
    filter_images,
    auto_split_dataset,
)

api_bp = Blueprint("api", __name__)

# -----------------------------
# ラベル保存
# -----------------------------
api_bp.add_url_rule("/labels", view_func=save_label, methods=["POST"])

# -----------------------------
# データセットエクスポート
# -----------------------------
api_bp.add_url_rule("/export_single/<int:image_id>", view_func=export_single_image_dataset, methods=["GET"])
api_bp.add_url_rule("/export/dataset", view_func=export_dataset, methods=["GET"])

# -----------------------------
# フィルタリング
# -----------------------------
api_bp.add_url_rule("/filter", view_func=filter_images, methods=["GET"])

# -----------------------------
# データセット自動分割
# -----------------------------
api_bp.add_url_rule("/auto-split", view_func=auto_split_dataset, methods=["POST"])
