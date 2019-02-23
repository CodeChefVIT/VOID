from flask import Flask
from flask import request
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
import keras
import pyaudio
import wave
from keras.models import model_from_json
import speech_recognition as sr

tf.keras.backend.clear_session()

json_file = open('model.json', 'r')
model_data = json_file.read()
json_file.close()
model = model_from_json(model_data)
model.load_weights("models/Emotion_Voice_Detection_Model.h5")
# print("Loaded model from disk")
global graph
graph = tf.get_default_graph()

def emotion_voice():
    X, sample_rate = librosa.load('output.wav',res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=0)
    feature = mfccs
    # print(feature.shape)
    X_test = np.array(feature)
    x_traincnn =np.expand_dims(X_test, axis=2)
    # print(np.array([x_traincnn]).shape)
    with graph.as_default():
        preds = model.predict(np.array([x_traincnn]),batch_size=32,verbose=1)
    # print(preds)
    preds1=preds.argmax(axis=1)
    mapper = ["angry_fe","calm_fe","fearful_fe","happy_fe","sad_fe","angry_m","calm_m","fearful_m","happy_m","sad_m"]
    return (mapper[preds1.tolist()[0]])

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def rec():
    r = sr.Recognizer()
    harvard = sr.AudioFile('output.wav')
    with harvard as source:
        audio = r.record(source)
    return(r.recognize_google(audio))

@app.route('/')
def base():
    return "Base URL"

@app.route('/voice', methods=['POST'])
def voice():
	print("VOICE")
	byte = base64.b64decode(request.form["data"].split(",")[1])
	newFile = open("output.wav", "wb")
	newFile.write(byte)
	newFile.close()
	return emotion_voice()

@app.route('/speech', methods=['GET'])
def speech():
    print("SPEECH-TO-TEXT")
    res = rec()
    # print(res)
    return res
if __name__ == '__main__':
    app.run(debug=True,port=5001)