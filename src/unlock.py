import numpy as np
import cv2

#cascade "profile"
FACE_CASCADE = cv2.CascadeClassifier("/home/leo/Documents/Projects/FacialRecognition_Directory/ressources/data/haarcascade_frontalface_default.xml")
R_EYE_CASACADE = cv2.CascadeClassifier("/home/leo/Documents/Projects/FacialRecognition_Directory/ressources/data/haarcascade_righteye_2splits.xml")
L_EYE_CASACADE = cv2.CascadeClassifier("/home/leo/Documents/Projects/FacialRecognition_Directory/ressources/data/haarcascade_lefteye_2splits.xml")

#defining the VideoCapture object
cap = cv2.VideoCapture(0)

while (True):
    #getting a frame
    ret, frame = cap.read()

    #changing color to gray for later process
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = FACE_CASCADE.detectMultiScale(frame_gray, scaleFactor=3, minNeighbors=3) #3,3 works the best for now
    for (x, y, w, h) in face:
        print(x, y, w, h)

        #drawing rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), ) #red rectangle

        Reyes = R_EYE_CASACADE.detectMultiScale(frame_gray, scaleFactor=3, minNeighbors=3)
        for (rx,ry,rw,rh) in Reyes:
            cv2.rectangle(frame,(rx+5,ry+5),(rx+rw-5,ry+rh-5),(255,0,0),2)
        Leyes = L_EYE_CASACADE.detectMultiScale(frame_gray, scaleFactor=3, minNeighbors=3)
        for (lx,ly,lw,lh) in Leyes:
            cv2.rectangle(frame,(lx+5,ly+5),(lx+lw-5,ly+lh-5),(255,0,0),2)


    #Displaying the frame (live video)
    cv2.imshow('LiveCapture', cv2.flip(frame, 1))

    #Checks if the q key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#releasing the VideoCapture object and closing the video feed
cap.release()
cv2.destroyAllWindows()
