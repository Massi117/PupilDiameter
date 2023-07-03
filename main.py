# This script extracts the frame of the video and calculates the size of the pupil

# Imports
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

# Capture the video
vidcap = cv.VideoCapture('run1.avi')
success,image = vidcap.read()
count = 0
plt.imshow(image)

while success:
    # Save frame as JPEG file
    #cv.imwrite("frame%d.jpg" % count, image)

    # Determine if a marker frame
    # Do later

    # Parse out eye image & convert to grayscale
    eye_image = image[130:350, 350:550]	
    eye_image = eye_image[:,:,0]
    #eye_image = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)

    # Apply a median filter (normalization)
    kernel = np.ones((4,4),np.float32)/16
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 210
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Canny Edge Detection
    eye_image = cv.Canny(eye_image,10,200)

    # Display the resulting frame
    cv.imshow('Frame', eye_image)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

    # Get the next frame
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

