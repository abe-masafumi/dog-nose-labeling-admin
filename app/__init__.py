from flask import Flask, send_from_directory
import os

def create_app():
    # static フォルダは front 用
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # images フォルダを静的配信するルートを追加
    @app.route('/images/<path:filename>')
    def serve_images(filename):
        images_dir = os.path.join(os.getcwd(), "images")  # プロジェクト直下の images/
        return send_from_directory(images_dir, filename)

    # Blueprint 登録
    from .routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    from .routes.ui import ui_bp
    app.register_blueprint(ui_bp)

    return app
