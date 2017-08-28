# coding=UTF-8
from flask import Flask, jsonify, request
from flask_api import status
from main import *

app = Flask(__name__)


@app.route('/getObjects', methods=["POST"])
def get_objects():

    upload_folder = "test_images"
    try:
        imagefile = request.files.get('imagefile')

        if imagefile:
            imagefile.save(os.path.join(upload_folder, imagefile.filename))
            objs = detect_objects(imagefile.filename, 0.50)
            os.remove(os.path.join(upload_folder, imagefile.filename))
            return jsonify(objs)

    except Exception as err:
        print(err.args[0])
        return jsonify(result=err.args[0]), status.HTTP_400_BAD_REQUEST


port = int(os.getenv('PORT', 8080))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
