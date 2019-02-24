import cv2
import numpy as np
from keras.models import load_model
from statistics import mode


USE_WEBCAM = True 
emotion_model_path = './models/data.hdf5'
emotion_labels = get_labels('fer2013')

frame_window = 10
emotion_offsets = (20, 40)

face_cascade = cv2.CascadeClassifier('./models/data.xml')
classifier = load_model(emotion_model_path)

emotion_target_size = classifier.input_shape[1:3]

emotion_window = []


cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) 
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') 

while cap.isOpened(): 
    ret, bgr_image = cap.read()


    grayImage = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = cascade.ScaleDetection(grayImage, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        face = gray_image[y1:y2, x1:x2]
        try:
            face = cv2.resize(face, (emotion_target_size))
        except:
            continue

        face = preprocess_input(face, True)
        face = np.expand_dims(face, 0)
        face = np.expand_dims(face, -1)
        emotion_prediction = emotion_classifier.predict(face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        print  ("emotion_text", emotion_text)
        
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
