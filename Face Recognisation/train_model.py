import cv2
import os
from PIL import Image
import numpy as np

#############
# To take imgs for dataset
def take_img():
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    ID = str(input("Enter ID"))
    Name = str(input("Enter Name"))
    sampleNum = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("dataset/" + Name + "." + ID + "." + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2.imshow("Frame",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum > 200:
            break
    cam.release()
    cv2.destroyAllWindows()

#To train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    global faces, Id
    try:
        faces, Id = getImagesAndLabels("dataset")
    except Exception as e:
        print("Please Make Dataset Folder & Put Images")

    #recognizer.train(faces,np.array(Id))
    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("model/trained_model2.yml")
        print("Model trained")
    except:
        print("Please make 'Model' Folder")

#To get imgs and id from dataset folder
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    # create empty face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage,'uint8')
        #getting id from img
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces =detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

############
take_img()
trainimg()
############
