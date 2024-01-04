# This script extracts the frame of the video and calculates the size of the pupil

# Imports
import cv2 as cv
import numpy as np
from scipy.io import savemat
import tensorflow as tf
from tqdm import tqdm
import time

# Dependencies
import image_processing as ip

DIR = "video/run1.avi"
TYPE = 1

# Load the model
new_model = tf.keras.models.load_model('CNN_model/ES_v1.h5')

# Start capture of the video
vidcap = cv.VideoCapture(DIR)
success,image = vidcap.read()

# Get # of frames
N = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))-56

# Initilize
diameter = []
frames = []
record = False

print('INITIALIZATION COMPLETE')
print('_______________________')

# Start timer
start_timer = time.perf_counter()

# Run analysis
print('Processing:')
for count in tqdm(range(N)):

    if not record:
        record = ip.startScan(image, type=TYPE)

    if record:
        # Parse out eye image & convert to grayscale
        eye_image_color = ip.parseImage(image, type=TYPE)
        eye_image = eye_image_color[:,:,0]  # Take the blue channel

        # Preprocess each frame
        frame_resized = cv.resize(eye_image_color, (64,64), interpolation=cv.INTER_AREA)
        new_frame = tf.convert_to_tensor(frame_resized, dtype=tf.float32)
        frame_tensor = tf.image.convert_image_dtype(new_frame, dtype=tf.float32, saturate=False)
        frame_tensor = tf.expand_dims(frame_tensor, axis=0)

        # Predict each frame
        prediction = new_model(frame_tensor, training=False)
        if prediction[0][0] >= prediction[0][1]:
            eye_open = False
        else:
            eye_open = True

        # Image Processing
        processed_frame = ip.preprocessImage(eye_image, Gsize=5, threshold=150)

        if eye_open:

            x, y, d = ip.predictCircle(processed_frame)

            # Append new diameter & frame #
            if d is not None:
                diameter.append(d)
                frames.append(count)

            # Draw the circumference of the circle.
            #cv.circle(eye_image_color, (x, y), r, (255, 255, 0), 2)
                
            # Draw a small circle (of radius 1) to show the center.
            #cv.circle(eye_image_color, (x, y), 1, (255, 0, 255), 3)     

        else:
            diameter.append(0)
            frames.append(count)

    # Display the resulting frame
    #cv.imwrite("frames/frame%d.jpg" % count, image)
    #cv.imshow("Detected Circle", eye_image_color)
    #cv.waitKey(25)

    # Get the next frame
    success,image = vidcap.read()

# Save the data
vidcap.release()
print('Alalysis complete: saving data...')

frames = [x - frames[0] for x in frames]
savemat('analysis/data/frame_count.mat', {'frame_count':frames})
savemat('analysis/data/diameters.mat', {'diameters':diameter})

print('Data saved.')

finish_timer = time.perf_counter()
print(f'Finished in {round(finish_timer-start_timer, 2)} second(s)')
print('DONE')
