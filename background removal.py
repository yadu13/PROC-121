# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')

# resizing the mountain image as 640 X 480

cv2.resize(mountain,640,640)
while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])
        upper_bound = np.array([255,255,255])
        mask_1 = cv2.inRange( lower_bound, upper_bound)

        lower_bound = np.array([100,100,100])
        upper_bound = np.array([255,255,255])
        mask_2 = cv2.inRange( lower_bound, upper_bound)

        mask_1 = mask_1 + mask_2
        # thresholding image
        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # inverting the mask
        img2=cv2.bitwise_not(img1)
        # bitwise and operation to extract foreground / person
        img1=cv2.bitwise_and(frame,frame,mask=mask_2)
        img2=cv2.bitwise_and(mountain,mountain,mask=mask_1)
        # final image
        final_image=np.where(img1==0,img1,img2)
        # show it
        cv2.imshow('frame' , frame)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
