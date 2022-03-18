import os
from tkinter.tix import IMAGE, Tree
import cv2
from flask import Flask, render_template, jsonify, request
import numpy as np
import json
import uuid
from PIL import Image


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
    



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000, use_reloader=True, threaded=False)