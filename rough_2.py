from flask import Flask, request, jsonify

import werkzeug
import os

app = Flask(__name__)

@app.route('/hello', methods =['PUT'])
def home():
    image_file = request.files['images']
    body = request.get_json(force=True)
    text = body['input']
    filename = werkzeug.utils.secure_filename(image_file)
    path = "uploaded_images/"+filename
    image_file.save("./uploaded_images/"+filename)

    return jsonify({
        "message":text,

    })

if __name__=="__main__":
    app.run(debug=True, port=9090)
