# coding=UTF-8

import sys
import os

from flask import Flask, jsonify, request, json, render_template, send_from_directory
from flask_api import status
from flask_cors import CORS, cross_origin
app = Flask(__name__)
import requests
from main import *
from decimal import Decimal



app = Flask(__name__)

@app.route('/getTest', methods=["GET"])
def getTest():
    return jsonify(result="bla")

@app.route('/getObjects', methods=["POST"])
def getObjects():
    #UPLOAD_FOLDER = "/Users/brunocf/Documents/General/CIO/Cognitive-Academy/TensorFlow/object_detection/test_images"
    UPLOAD_FOLDER = "test_images"
    try:
        imagefile = request.files.get('imagefile')

        if imagefile:
            #filename = secure_filename(imagefile.filename)
            imagefile.save(os.path.join(UPLOAD_FOLDER, imagefile.filename))
            objs = detect_objects(imagefile.filename, 0.50)
            os.remove(os.path.join(UPLOAD_FOLDER, imagefile.filename))
            #print(str(objs))
            return jsonify(objs)

    except Exception as err:
        print(err.args[0])
        return jsonify(result=err.args[0]),status.HTTP_400_BAD_REQUEST


port = int(os.getenv('PORT', 8080))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
