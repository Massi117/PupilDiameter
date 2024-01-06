# Image analysis functions

# Imports
import cv2 as cv
import numpy as np
import math

def getType(image):
    '''
    returns the type of video format
    '''
    height, width , channels = image.shape
    if height < 200:
        return 2
    else:
        return 1

def startScan(image, type):
    '''
    returns true if the frame indicates a start of the scan
    '''
    if type == 1:
        pixel = image[10][450]
        if (pixel == [102, 102, 100]).all():
            return True
        else:
            return False
    elif type == 2:
        pixel = image[115][210]
        if (pixel <= [5, 5, 5]).all():
            return False
        else:
            return True
    else:
        raise Exception("parameter 'type' must be a 1 or 2")


def parseImage(image, type=1):
    '''
    returns a parsed image
    '''
    # Parse out eye image
    if type == 1:
        parsed_image = image[130:350, 350:550]
    elif type == 2:
        parsed_image = image[20:115, 5:100]
    else:
        raise Exception("parameter 'type' must be a 1 or 2")
    
    return parsed_image

def preprocessImage(image, Gsize, threshold):
    '''
    returns a preprocessed image
    '''
    # Apply a median filter (normalization)
    kernel = np.ones((Gsize,Gsize),np.float32)/(Gsize**2)
    preprocessedImage = cv.filter2D(image,-1,kernel)

    # Binarize the image
    sucess, preprocessedImage = cv.threshold(preprocessedImage,threshold,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    if not sucess:
        raise Exception("image could not be binarised")

    return preprocessedImage

def predictCircle(image, type):
    '''
    returns radius and location of detected circle
    '''
    if type == 1:
        max = 40
        min = 5
    elif type ==2:
        max = 14
        min = 5
    else:
        raise Exception("parameter 'type' must be a 1 or 2")
    
    # Apply Hough Transform
    detected_circles = cv.HoughCircles(image,
                                       cv.HOUGH_GRADIENT,
                                       1,
                                       1000,
                                       param1=450,
                                       param2=1,
                                       minRadius=min,
                                       maxRadius=max)

    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
            
        for pt in detected_circles[0, :]: 
            x, y, r = pt[0], pt[1], pt[2]
            d = r*2

        return x, y, d
    
    else:
        return None, None, None


def detectOutlier(x, y, d, centerX, centerY, prevD):
    '''
    returns True if the center of the circle is on average
    far away than previous iterations
    '''
    distanceX = abs(x-centerX)
    distanceY = abs(y-centerY)
    if distanceX > d or distanceY > d or (abs(d-prevD)>10):
        return True
    else:
        return False