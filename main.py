# This script extracts the frame of the video and calculates the size of the pupil

# Imports
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow as tf

# Import our model
from build_model import build_cnn_model

# Define the model
new_model = build_cnn_model()
# Initialize the model by passing some data through
new_model.build(input_shape=(64,64,3,1))
# Print the summary of the layers in the model.
print(new_model.summary())

new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam', metrics = ['accuracy'])

# Loads the weights
new_model.load_weights("eyedetection.h5")

# Start capture of the video
vidcap = cv.VideoCapture('run1.avi')
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
    eye_image_color = image[130:350, 350:550]	
    eye_image = eye_image_color[:,:,2]
    #eye_image = cv.cvtColor(eye_image_color, cv.COLOR_BGR2GRAY)

    # Preprocess each frame
    frame_resized = cv.resize(eye_image_color, (64,64), interpolation=cv.INTER_AREA)
    #grayFrame = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    new_frame = tf.convert_to_tensor(frame_resized, dtype=tf.float32)
    frame_tensor = tf.image.convert_image_dtype(new_frame, dtype=tf.float32, saturate=False)
    frame_tensor = tf.expand_dims(frame_tensor, axis=0)

    # Predict each frame
    prediction = new_model.predict(frame_tensor)
    if prediction[0][0] >= prediction[0][1]:
        print('Single Image Prediction: Closed Eyes')
        eye_open = False
    else:
        print('Single Image Prediction: Open Eyes')
        eye_open = True

    # Apply a median filter (normalization)
    l = 15
    kernel = np.ones((l,l),np.float32)/(l**2)
    eye_image = cv.filter2D(eye_image,-1,kernel)

    # Binarize the image
    #thresh = 100
    #ret,eye_image = cv.threshold(eye_image,thresh,255,cv.THRESH_BINARY_INV)

    # Canny Edge Detection
    #eye_image = cv.Canny(eye_image,10,200)

    # Hough Transform
    detected_circles = cv.HoughCircles(eye_image,cv.HOUGH_GRADIENT,1,100,param1=50,param2=20,minRadius=20,maxRadius=40)

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
    
    #else:

      #print('No circles detected')
      #cv.imshow('Frame', eye_image)

    # Display the resulting frame
    cv.imshow("Detected Circle", eye_image_color)
    cv.waitKey(25)

    # Get the next frame
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1

# Plot the relative diameter of the pupil vs frame
plt.clf()
plt.plot(frames, diameter)
plt.ylim(0, 40)
plt.show()

