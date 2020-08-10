import cv2
import numpy as np

# pip install opencv-contrib-python

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/trained_model2.yml")
faceCascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 260, 0), 7)
        cv2.putText(img, str(Id), (x, y-40), font, 2, (255, 255, 255), 3)
    cv2.imshow("image",img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()