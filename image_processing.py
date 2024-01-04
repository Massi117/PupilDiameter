# Image analysis functions

# Imports
import cv2 as cv
import numpy as np

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
        pixel = image[0][0]
        if (pixel == [102, 102, 100]).all():
            return True
        else:
            return False
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
        parsed_image = image[130:350, 350:550]
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
    sucess, preprocessedImage = cv.threshold(preprocessedImage,threshold,255,cv.THRESH_BINARY)

    if not sucess:
        raise Exception("image could not be binarised")

    return preprocessedImage

def predictCircle(image):
    '''
    returns radius and location of detected circle
    '''
    # Apply Hough Transform
    detected_circles = cv.HoughCircles(image,
                                       cv.HOUGH_GRADIENT,
                                       1,
                                       1000,
                                       param1=450,
                                       param2=1,
                                       minRadius=5,
                                       maxRadius=40)

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

