# Pupil Dialation Measure (v1.0.0)

A set of scripts that meaure the dialation of the pupil from video captured in conjunction with an fMRI scan.

## Getting Started

To load the project, clone the git repository onto your PC.

### Prerequisites

The things you need before installing the software.

* Python 3.8.10
* git bash
* vscode (or any text editor)

### Installation

Once the repo is cloned onto your PC, git bash onto the directory with the env/ folder and activate the python virtual environment as follows.

```
$ cd env/Scripts
$ . activate
$ cd ..
$ cd ..
```
Once activated, download the required dependencies.

```
$ pip install -r requirements.txt
```

## Usage

The script main.py measures the pupil diameter of every frame of a video, to use it, just upload an mp4 or avi video into the same directory and update the name and extention in the script.

```
$ vidcap = cv.VideoCapture('myVideo.avi')
```
The build_history.py scripts builds the plot of the eye diamter over each frame from the saved data.

## How it works

A CNN trained on a dataset of 14,000 frames is used to determine when the eye is open or closed. When open, the blue channel (in this case denoting the infrared) of image is passed through a gaussian kernel. 

Canny edge detection is then used to map out the major edges of the image, which allows the use of the Hough Transform to determine the likely center and radius of the pupil.

## Acknowledgments

Created by Massinissa Bosli & Stephanie Williams