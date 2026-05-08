from src.web_applications.unit_converter.webapp import create_app
import traceback
try:
    app = create_app()
    app.run(port=5000)
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
