# This script plots the eye data

# Imports
import pickle
from matplotlib import pyplot as plt

# Load the data
with open("eye_frames", "rb") as fd:
    f = pickle.load(fd)

with open("eye_diameter", "rb") as dd:
    d = pickle.load(dd)

# Plot the relative diameter of the pupil vs frame
plt.clf()
plt.plot(f, d, 'o')

# Label axes
plt.xlabel("Frame #")
plt.ylabel("Eye diameter (pixels)")

# Set axes scaling
plt.ylim(0, 50)

# Show the plot
plt.show()