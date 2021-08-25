import cv2
import face_recognition
import numpy as np
import os
import datetime

# Trained dateset for faces
trainedDatasetFaces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

folderName = "images"
images = []
names = []

imgList = os.listdir(folderName)
# print(imgList)  # imgList = ['Bill Gates.jpg', 'Elon Musk.jpg', 'RDJ.jpg']

#  Loading Images
for faceLocation in imgList:
    currentImg = cv2.imread(f"{folderName}/{faceLocation}")
    images.append(currentImg)
    names.append(os.path.splitext(faceLocation)[0])
# print(names)  # names = ['Bill Gates', 'Elon Musk', 'RDJ']

# Finding Encodings
def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodingListKnown = findEncodings(images)
print("Encodings Completed")

# Adding the names to attendance list
def attendance(name):
    with open("Attendance.csv", "r+") as f:
        attendanceList = f.readlines()
        nameList = []
        for fileContents in attendanceList:
            entry = fileContents.split(",")
            nameList.append(entry[0])
        # Adding name to the attendance list if it's not presented
        if name not in nameList:
            time = datetime.datetime.now().strftime("%H:%M:%S %p")
            date = datetime.datetime.now().strftime("%d-%b-%y")
            f.writelines([f"\n{name},{date},{time}"])

# Capturing video from webcam
video = cv2.VideoCapture(0)
while True:
    success, frame = video.read()

    # Converting colored img to gray img
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Getting locations of the face to draw a rectangle
    faceLocations = face_recognition.face_locations(frame)
    # Getting x, y, w, h of the faces
    faces = trainedDatasetFaces.detectMultiScale(grayFrame)
    # print(faces)
    faceEncodings = face_recognition.face_encodings(frame, faceLocations)

    for face, faceEncoding in zip(faces, faceEncodings):

        faceResult = face_recognition.compare_faces(encodingListKnown, faceEncoding)
        faceDistance = face_recognition.face_distance(encodingListKnown, faceEncoding)
        print(faceResult, faceDistance)
        # print(type(faceDistance))

        faceResultIndex = np.argmin(faceDistance)
        # print(faceResultIndex)

        if faceResult[faceResultIndex]:
            name = names[faceResultIndex].upper()
            attendance(name)
            print(name)

            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(frame, (x, y-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Web Cam", frame)
        cv2.waitKey(1)