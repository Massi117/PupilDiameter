# This script extracts the frame of the video and calculates the size of the pupil

# Imports
import cv2 as cv
import numpy as np
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt

# Import our model
#from CNN_model.build_model import build_cnn_model

# Load the model
new_model = tf.keras.models.load_model('CNN_model/ES_v1.h5')

# Start capture of the video
vidcap = cv.VideoCapture('video/run2.avi')
success,image = vidcap.read()
count = 0
plt.imshow(image)

# Initilize diameter & frame list
diameter = []
frames = []

while success:
    # Save frame as JPEG file
    #cv.imwrite("frame%d.jpg" % count, image)

    # Determine if a marker frame
    # Do later

    # Parse out eye image & convert to grayscale
    eye_image_color = image[40:100, 20:80]	
    eye_image = eye_image_color[:,:,0]
    #eye_image = cv.cvtColor(eye_image_color, cv.COLOR_BGR2GRAY)

    # Preprocess each frame
    frame_resized = cv.resize(eye_image_color, (64,64), interpolation=cv.INTER_AREA)
    #grayFrame = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    new_frame = tf.convert_to_tensor(frame_resized, dtype=tf.float32)
    frame_tensor = tf.image.convert_image_dtype(new_frame, dtype=tf.float32, saturate=False)
    frame_tensor = tf.expand_dims(frame_tensor, axis=0)

    # Predict each frame
    prediction = new_model.predict(frame_tensor,verbose=0)
    if prediction[0][0] >= prediction[0][1]:
        #print('Single Image Prediction: Closed Eyes')
        eye_open = False
    else:
        #print('Single Image Prediction: Open Eyes')
        eye_open = True

    # Binarize the image
    thresh = 200
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_TOZERO)

    # Apply a median filter (normalization)
    l = 5
    kernel = np.ones((l,l),np.float32)/(l**2)
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 115
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Apply a median filter (normalization)
    l = 10
    kernel = np.ones((l,l),np.float32)/(l**2)
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 50
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Apply a median filter (normalization)
    l = 10
    kernel = np.ones((l,l),np.float32)/(l**2)
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    thresh = 70
    ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY)

    # Canny Edge Detection
    #eye_image = cv.Canny(eye_image,10,200)

    if eye_open:

        # Hough Transform
        detected_circles = cv.HoughCircles(eye_image,cv.HOUGH_GRADIENT,1,100,param1=450,param2=1,minRadius=4,maxRadius=30)

        # Draw circles that are detected.
        if detected_circles is not None:
    
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            
            for pt in detected_circles[0, :]:
                
                a, b, r = pt[0], pt[1], pt[2]

                # Append new diameter & frame #
                diameter.append(r)
                frames.append(count)

                # Draw the circumference of the circle.
                cv.circle(eye_image_color, (a, b), r, (0, 255, 0), 2)
            
                # Draw a small circle (of radius 1) to show the center.
                cv.circle(eye_image_color, (a, b), 1, (0, 0, 255), 3)

        else:
            # Append new diameter & frame #
            diameter.append(0)
            frames.append(count)

            print('Pupil not detected in frame #', count)
    else:

        diameter.append(0)
        frames.append(count)

    # Display the resulting frame
    cv.imshow("Detected Circle", eye_image_color)
    cv.waitKey(25)

    # Get the next frame
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1

# Save the data
with open("eye_frames", "wb") as fd:
    pickle.dump(frames, fd)

with open("eye_diameter", "wb") as dd:
    pickle.dump(diameter, dd)