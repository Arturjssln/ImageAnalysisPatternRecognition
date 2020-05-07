import cv2
import numpy as np

MIN_CONTOUR_POINT = 20

def find_contour(img, opencv_version):
    """ Finds and returns the contour of the image"""
    contour = []
    if int(opencv_version) == 3:
        _, contour, _ = cv2.findContours(img, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
    else:
        contour, _ = cv2.findContours(img.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
    
    contour_array = contour[0].reshape(-1, 2)
    # minimum contour size required
    if contour_array.shape[0] < MIN_CONTOUR_POINT:
        contour_array = contour[1].reshape(-1, 2)
    return contour_array

def convert_contour(contour):
    contour_complex = np.empty(contour.shape[:-1], dtype=complex)
    contour_complex.real = contour[:, 0]
    contour_complex.imag = contour[:, 1]
    return contour_complex

def find_descriptor(contour):
    """ Finds and returns the Fourier-Descriptor from the image contour"""
    return np.fft.fft(contour)