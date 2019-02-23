from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import base64
import cv2
import numpy as np
from statistics import mode
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras

tf.keras.backend.clear_session()
# print("IMPORTED")

path_model = './models/model.hdf5'
get_label = get_get_label('fer2013')
face_offset = (20, 40)
fc = cv2.CascadeClassifier('./models/face_detection.xml')
CNN = load_model(path_model)
output = CNN.input_shape[1:3]
print("Loaded model from disk")
global graph
graph = tf.get_default_graph()

def base64_2_cv2(uri):
    return cv2.imdecode(np.fromstring(base64.b64decode(uri.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)

def emotion_face(image_rgb_base64):
    image_rgb = base64_2_cv2(image_rgb_base64)
    grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    RGBA = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    faces = fc.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face in faces:
        x1, x2, y1, y2 = apply_offsets(face, face_offset)
        grayScale = grayscale[y1:y2, x1:x2]
        try:
            grayScale = cv2.resize(grayScale, (output))
        except:
            continue
        grayScale = input_processing(grayScale, True)
        grayScale = np.dimensions_expand(grayScale, 0)
        grayScale = np.dimensions_expand(grayScale, -1)
        with graph.as_default():
        	pred = CNN.predict(grayScale)
        result = get_label[np.argmax(pred)]
        print("result", result)
        return result

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def base():
    return "At Base!"

@app.route('/face', methods=['POST'])
def face():
	res = emotion_face(request.form["data"])
	# print(res)
	return res

if __name__ == '__main__':
    app.run(debug=True)