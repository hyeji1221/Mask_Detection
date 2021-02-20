import cv2 # pip install --upgrade opencv-contrib-python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


model = 'Face_recognition\opencv_face_detector_uint8.pb'
config = 'Face_recognition\opencv_face_detector.pbtxt'
#mask_model = tf.keras.models.load_model('./Final_model.h5')
#mask_model = tf.keras.models.load_model('./BaeEungi/Final_model.h5') # 임시 모델 사용
mask_model = tf.keras.models.load_model('./mask_model.h5')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened(): # 예외처리
    print('Camera open failed!')
    exit()

net = cv2.dnn.readNet(model, config)

if net.empty(): # 예외처리
    print('Net open failed!')
    exit()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if frame is None:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            continue

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        face = frame[y1:y2, x1:x2] # 크롭
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        image_w = 224
        image_h = 224
        fx = image_w / face.shape[1]
        fy = image_h / face.shape[0]
        img = cv2.resize(face, None, fx=image_w / face.shape[1], fy=image_h / face.shape[0])
        img = (img / 256)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        """
        face = cv2.resize(face, (28, 28), 3)
        face = face / 256
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0) """
        result = mask_model.predict(img).squeeze()
        print(mask_model.predict(img).squeeze())
        mask = result[0]
        withoutMask = result[1]
        #(mask, withoutMask) = model.predict(face)[0]

        if (mask > withoutMask):
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (withoutMask * 100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color) # 사각형 그리기

        label = 'Face: %s' % label
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27: # ESC
        break

cv2.destroyAllWindows()
