import numpy as np
import cv2
import pickle

face_cc = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')      #here we bring in our trained data model to further run predictions on the data items

labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)       #here we load the pickle data to actually get a name of the image that is detected by the vidframe
    labels = {v:k for k,v in og_labels.items()}     #k=key,v=value


cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read() #frame by frame capturing of video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cc.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]      #roi= region of interest

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_PLAIN
            name = labels[id_]      #this is the actual text that will be displayed
            color = (255, 255, 255)     #color of the text appearing over the rectangle border
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 2, color, stroke, cv2.LINE_AA)

        img_item = 'myimage.png'
        cv2.imwrite(img_item, roi_gray)     #cv2.imwrite(img_item, roi_frame) it will display colored photo

        color = (255, 0, 0)     #color of the rectangle around face; format is BGR; range is 0-255
        stroke = 2          #thickness of rectangle border

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)    #this draws a rectangle around the face in video frame with specified width and height

    cv2.imshow('vidframe', frame) #display the frame if true

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q on keyboard to close the video frame
        break #this breaks the infinite video capture loop by clicking on q 

cap.release()
cv2.destroyAllWindows()