# import tensorflow.keras.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
# from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import cv2
import numpy as np
# import tensorflow.keras.utils.load_img
import warnings
import dlib
warnings.filterwarnings("ignore")

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

while True:
    # captures frame and returns boolean value and captured image
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    # RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Iterator to count faces
    i = 0
    for face in faces:

        # Get the coordinates of faces
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Increment iterator for each face in faces
        i = i+1

        # Display the box and faces
        cv2.putText(frame, 'Faces : '+str(i), (x-100, y-100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(face, i)

    # # Display the resulting frame
    # cv2.imshow('frame', frame)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (255, 0, 0), thickness=7)
        # cropping region of interest i.e. face area from  image
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy',
                    'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, 'Expression : ' + predicted_emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
