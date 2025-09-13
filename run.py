from app import create_app
import sys
from app.services.detection_service import detect_nose_for_all_images
from app.db import init_db, register_images
import os


app = create_app()
print(">>> static folder is:", os.path.abspath(app.static_folder))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "detect_nose":
        detect_nose_for_all_images()
    else:
        init_db()
        register_images()
        app.run(debug=True, host="0.0.0.0", port=8080)
