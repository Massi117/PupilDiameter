# This script plots the eye data

# Imports
import pickle
from scipy.io import savemat

# Load the data
with open("data/eye_frames", "rb") as fd:
    f = pickle.load(fd)

with open("data/eye_diameter", "rb") as dd:
    d = pickle.load(dd)

# Save the data to .mat files

# Frames
mdict = {"f": f, "label": "experiment"}
savemat('data/frames.mat', mdict)

# radi
mdict = {"d": d, "label": "experiment"}
savemat('data/diameter.mat', mdict)
