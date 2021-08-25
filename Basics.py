import cv2
import face_recognition
import numpy as np

# known image
imgElon = face_recognition.load_image_file("images/Elon Musk.jpg")  # imgElon=cv2.imread("images/Elon Musk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Test image
imgElonTest = face_recognition.load_image_file("test images/elon test.jpg")  # imgElon=cv2.imread("test images/elon test.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# known image location & encodings
faceLoc = face_recognition.face_locations(imgElon)[0]  # returns tuple of TOP, RIGHT, BOTTOM, LEFT values
encodingsElon = face_recognition.face_encodings(imgElon)[0]
# print(encodingsElon, len(encodingsElon))
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)

# Test image location & encodings
faceLocTest = face_recognition.face_locations(imgElonTest)[0]  # returns tuple of TOP, RIGHT, BOTTOM, LEFT values
encodingsElonTest = face_recognition.face_encodings(imgElonTest)[0]
# print(encodingsElonTest, len(encodingsElonTest))
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 0), 2)

faceResults = face_recognition.compare_faces([encodingsElon], encodingsElonTest)  # returns Boolean
faceDistance = face_recognition.face_distance([encodingsElon], encodingsElonTest)
# print(faceResults, faceDistance)

cv2.putText(imgElonTest, f"{faceResults[0]}, {round(faceDistance[0], 2)}", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2)

# cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Elon Test", imgElonTest)
cv2.waitKey(0)
