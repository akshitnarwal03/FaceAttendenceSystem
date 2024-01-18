import pandas as pd
import cv2
import numpy as np
import os
import face_recognition
import time
import pickle  # Import the pickle module

path = 'image folder path'

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (len(face_recognition.face_encodings(img)) != 0 ):
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print(f"{len(encodeList)} done , {len(images) - len(encodeList)} left")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Store the encoded list in a pickle file
with open('file_name.pkl', 'wb') as f:
    pickle.dump(encodeListKnown, f)
print('Encoding List Stored in encodings.pkl')
