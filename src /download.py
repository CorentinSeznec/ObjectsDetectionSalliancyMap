import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time

def create_frame(path_tovideo):
    cap = cv.VideoCapture(path_tovideo)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    i = 0

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            cv.imwrite('/home/onajib/Analyse vidéos/Frames_Bounding/frame'+str(i)+'.jpg',frame)
            i = i+1
            print(i)

        # # Display the resulting frame
        #     cv.imshow('Frame',frame)
        #     plt.show()
        
        # Press Q on keyboard to  exit
        # if cv.waitKey(25) & 0xFF == ord('q'):
        #     break
        # Break the loop
        else: 
            break
