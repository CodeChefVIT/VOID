from flask import Flask
from flask import request, render_template
from flask_cors import CORS, cross_origin
import base64
import cv2
import numpy as np
from statistics import mode
from keras.models import load_model
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing.text import Tokenizer
import pyaudio
import wave
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

tf.keras.backend.clear_session()
print("IMPORTED")
path = './models/data.hdf5'
emotion_labels = get_labels('fer2013')
frame_window = 10
emotion_offsets = (20, 40)
face_cascade = cv2.CascadeClassifier('./models/data.xml')
classifier = load_model(path)
emotion_target_size = classifier.input_shape[1:3]
emotion_window = []
print("Loaded model from disk")
global graph
graph = tf.get_default_graph()

def data_uri_to_cv2_img(uri):
    data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def emotion(bgr_image_base64):
    bgr_image = data_uri_to_cv2_img(bgr_image_base64)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        with graph.as_default():
        	emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        print("emotion_text", emotion_text)
        return emotion_text

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
    return "HELLO"

@app.route('/face', methods=['POST'])
def face():
	res = emotion(request.form["data"])
	print(res)
	return res

if __name__ == '__main__':
    app.run(debug=True)