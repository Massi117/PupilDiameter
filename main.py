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

DIR = "/ad/eng/research/eng_research_lewislab/data/emot/mood35sd/eyes/run1.avi"

# Load the model
new_model = tf.keras.models.load_model('CNN_model/ES_v1.h5')

# Start capture of the video
vidcap = cv.VideoCapture(DIR)
success,image = vidcap.read()
TYPE = ip.getType(image)

# Get # of frames
N = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))-56

# Initilize
diameter = []
frames = []
centerX = centerY = None
d = None
record = True
outlier = False

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
        processed_frame = ip.preprocessImage(eye_image, Gsize=5, threshold=200)

        if eye_open:

            x, y, d = ip.predictCircle(processed_frame, TYPE)

            # Detect outliers
            if centerX is None:
                height, width, chanels = eye_image_color.shape
                centerX = width/2
                centerY = height/2
                prevD = d

            outlier = ip.detectOutlier(x, y, d, centerX, centerY, prevD) 

            if outlier:
                # Get the next frame
                success,image = vidcap.read()
                continue
            
            prevD = d

            diameter.append(int(d))
            frames.append(count)
            
            # Draw the circumference of the circle.
            #cv.circle(eye_image_color, (x, y), int(d/2), (255, 255, 0), 2)
                
            # Draw a small circle (of radius 1) to show the center.
            #cv.circle(eye_image_color, (x, y), 1, (255, 0, 255), 3)     

        else:
            diameter.append(0)
            frames.append(count)

        # Save & display the resulting frame
        #cv.imwrite("frames/frame%d.jpg" % count, eye_image_color)
        #cv.imshow("Detected Circle", eye_image_color)
        #cv.waitKey(25)

    # Get the next frame
    success,image = vidcap.read()

# Throw error if no start point exists:
if not record:
    raise Exception("No start indication for this run")

# Save the data
vidcap.release()
print('Alalysis complete: saving data...')

frames = [x - frames[0] for x in frames]
dict = {"frames": frames, "diameters": diameter}
savemat('analysis/data/data_dict.mat', dict)

print('Data saved.')

finish_timer = time.perf_counter()
print(f'Finished in {round(finish_timer-start_timer, 2)} second(s)')
print('DONE')
