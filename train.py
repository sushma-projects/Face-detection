import os
import cv2
import numpy as np
from PIL import Image       #pil = python image library
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0      #this is the number value associated with the labels defined below
label_ids = {}      #this is an empty dictionary defined for storing the ids of the labels
y_labels = []
x_train = []        #this has the actual number of pixel values

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('png'):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            #print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id       #if label belongs in the label_ids dictionary, then we assign that value to current_id
                current_id = current_id+1       #value of current_id is incremented by 1
            
            id_ = label_ids[label]      #the variable id_ is given that value associated with label in the dictionary
            #print(label_ids)

            #y_labels.append(label)
            #x_train.append(path)

            pil_image = Image.open(path).convert('L')   #this turns image into GRAY
            image_array = np.array(pil_image, 'uint8')      #converts image into numpy array
            #print(image_array)
            faces = face_cc.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5) 
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


#print(y_labels)
#print(x_train)


with open('labels.pickle', 'wb') as f:      #labels.pickle is the name of pickle items; f is the file
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
