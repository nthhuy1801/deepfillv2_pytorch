import base64
import os
import cv2
from flask import Flask, render_template, jsonify, request
import numpy as np
import json
import uuid
from PIL import Image
from predict import predict


UPLOAD_FOLDER  = 'static/uploads'
ALLOWED_EXTENSIONS =  set(['png', 'jpg', 'jpeg'])

# Config Flask
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["CACHE_TYPE"] = "null"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index_all():
    return render_template('index.html')

def resize(img_path, size=(128,128)):
    img = Image.open(img_path)
    img = img.resize(img, Image.ANTIALIAS)
    img.save(img_path)
    
@app.route('/process', methods=["POST"])
def process():
    try:
        filename = str(uuid.uuid4())
        file_path_raw = os.path.join(app.config["UPLOAD_FOLDER"], filename + '.png')
        file_path_mask = os.path.join(app.config["UPLOAD_FOLDER"], 'mask_' + filename + '.png')
        file_path_output = os.path.join(app.config["UPLOAD_FOLDER"], 'output_' + filename + '.png')

        # Save mask
        mask_b64 = request.values[('mask_b64')]
        print('Post success')
        img_str = mask_b64.split(',')[1]
        output = open(file_path_mask, 'wb')
        decoded = base64.b64decode(img_str)
        output.write(decoded)
        output.close()
        resize(file_path_mask)

        # Save raw image
        file_raw = request.files.get('input_file')
        file_raw.save(file_path_raw)
        resize(file_path_raw)

        # Doing inpainting
        output = predict(file_path_raw, file_path_mask)
        cv2.imwrite(file_path_output, output)
        resize(file_path_output, size=(256,256))

        return jsonify(
            {
                'output_image': os.path.join('static', 'uploads', os.path.basename(file_path_output))
            }
        )

    except Exception as error:
        print(error)
        return jsonify({'status': 'error'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, use_reloader=True, threaded=False)