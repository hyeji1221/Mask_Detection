import cv2

model = 'Face_recognition\opencv_face_detector_uint8.pb'
config = 'Face_recognition\opencv_face_detector.pbtxt'

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
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0)) # 사각형 그리기
       # frame = frame[y1 - int(h/4):y2 + h + int(h/4), x1 - int(w/4):x2 + w + int(w/4)] # 이미지 크롭

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27: # ESC
        break

cv2.destroyAllWindows()
