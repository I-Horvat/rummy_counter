from flask import Flask, request, jsonify
from PIL import Image
import io

from flask_cors import CORS

from testing.test_image import  process_image

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        points=process_image(image)
        return jsonify({"points": f"{points}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)