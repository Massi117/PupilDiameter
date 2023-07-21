# This script extracts the frame of the video and calculates the size of the pupil

# Imports
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt

# Start capture of the video
vidcap = cv.VideoCapture('video/run2.avi')
success,image = vidcap.read()
count = 0
plt.imshow(image)

while success:
    # Save frame as JPEG file
    #cv.imwrite("frame%d.jpg" % count, image)

    # Determine if a marker frame
    # Do later

    # Parse out eye image & convert to grayscale
    eye_image_color = image[40:100, 20:80]
    #eye_image_color = image[130:350, 350:550]	# normal
    eye_image = eye_image_color[:,:,0]
    #eye_image = cv.cvtColor(eye_image_color, cv.COLOR_BGR2GRAY)

    # Binarize the image
    thresh = 175
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_TOZERO)

    # Apply a median filter (normalization)
    l = 20
    kernel = np.ones((l,l),np.float32)/(l**2)
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 50
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Apply a median filter (normalization)
    l = 20
    kernel = np.ones((l,l),np.float32)/(l**2)
    #eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 10
    #ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Apply a median filter (normalization)
    l = 10
    kernel = np.ones((l,l),np.float32)/(l**2)
    #eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 70
    #ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Canny Edge Detection
    eye_image = cv.Canny(eye_image,225,450)

    if True:

        # Hough Transform
        detected_circles = cv.HoughCircles(eye_image,cv.HOUGH_GRADIENT,1,1000,param1=450,param2=1,minRadius=5,maxRadius=50)

        # Draw circles that are detected.
        if detected_circles is not None:
    
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            
            for pt in detected_circles[0, :]:
                
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                #cv.circle(eye_image_color, (a, b), r, (0, 255, 0), 2)
            
                # Draw a small circle (of radius 1) to show the center.
                #cv.circle(eye_image_color, (a, b), 1, (0, 0, 255), 3)


    # Display the resulting frame
    cv.imshow("Processed", eye_image)
    cv.imshow("Detected Circle", eye_image_color)
    cv.waitKey(25)

    # Save frame as JPEG file
    cv.imwrite("frame%d.jpg" % count, eye_image)

    # Get the next frame
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1