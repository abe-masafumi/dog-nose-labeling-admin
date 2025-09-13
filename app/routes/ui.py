from flask import Blueprint
from app.controllers.ui_controller import show_index, show_review, show_export

ui_bp = Blueprint("ui", __name__)
ui_bp.add_url_rule("/", view_func=show_index)
ui_bp.add_url_rule("/review", view_func=show_review)
ui_bp.add_url_rule("/export", view_func=show_export)
