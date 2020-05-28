"""
    Utility functions
"""
from skimage.color import rgb2gray
import numpy as np
import skimage.filters as filt
import cv2
import matplotlib.pyplot as plt
from itertools import product

def threshold(img, th1=None, th2=None):
    """
    Threshold function
    """
    out = np.ones(np.shape(img))
    if th1 is not None:
        out[np.where(img < th1)] = 0
    if th2 is not None:
        out[np.where(img > th2)] = 0
    return out

def check(list_, val_):
    """
    Function to check if a neighbor has label, if yes, it will return the label
    """
    if val_ > min(list_) and min(list_) > 0:
        return min(list_)
    return False


def labeling(img, label):
    """
    Return an image with element corresponding to the given label
    """
    out = np.zeros(np.shape(img), dtype=np.uint)
    out[np.where(img == label)] = 1
    return out


def coor_object(object_):
    """
    Return the coordinate of the center of an object
    """
    pos_x, pos_y = np.where(object_ != 0)
    
    return [pos_y.mean(), pos_x.mean()]

def extract_object(image, label):
    """
    Return the binary img, number of pixel and coordinate
    of the center of an object with a given label
    """
    out = np.zeros(np.shape(image))
    out[np.where(image == label)] = 1
    pxl = sum(sum(out))
    coor = coor_object(out)
    return out, pxl, coor


def find_objects(frame):
    """
    Return the coordinates of the digits, of the arrow and the average color of the arrow
    """
    # from RGB colors to gray scale
    image = rgb2gray(frame)
    ones = np.ones_like(image)
    # reshape : remove black lines on the side (out of table)
    image = np.delete(image, range(100), axis=1)
    image = np.delete(image, range(image.shape[1]-100, image.shape[1]), axis=1)
    # Threshold on image to separate background from foreground
    # We are using an adaptive threshold that find the best threshold depending of the illumination
    thres = filt.threshold_otsu(image)
    ones[:, 100:frame.shape[1]-100] = threshold(image, thres)
    image = ones
    new_label = 1
    im_h, im_w = image.shape
    label = np.zeros((im_h, im_w), dtype=np.uint)
    # Iterating on each pixels of the image
    nb_neighbors = 10
    for i in range(1, im_h - nb_neighbors):
        for j in range(1, im_w - nb_neighbors):
            # Labeling each pixels of the foreground with same value as its neighbor
            if image[i, j] == 0:
                label_neighbors = np.max([label[i+a, j+b] for a in range(-nb_neighbors, nb_neighbors+1)
                                          for b in range(- nb_neighbors, nb_neighbors + 1)])
                if label_neighbors != 0:
                    label[i, j] = label_neighbors
                else:
                    label[i, j] = new_label + 1
                    new_label += 1

    # Merging of labels for objects that are containing different labels
    nb_neighbors2 = 2
    for i in range(1, im_h - nb_neighbors2):
        for j in range(1, im_w - nb_neighbors2):
            label_neighbors = [label[i+a, j+b] for a in range(-nb_neighbors2, nb_neighbors2+1)
                               for b in range(- nb_neighbors2, nb_neighbors2 + 1)]
            val_min = check(label_neighbors, label[i, j])
            if check(label_neighbors, label[i, j]):
                label[np.where(label == label[i, j])] = val_min
    
    label_unique = np.unique(label)
    # Background is not an object so we subtract 1
    nb_objects = len(label_unique) - 1
    # Calculating number of pixel per object and the position of the center
    nb_pixels = []
    avg_coor = []
    for label_choice in range(1, nb_objects+1):
        _, pxl, coor = extract_object(label, label_unique[label_choice])
        # Digits
        if pxl < 300: 
            avg_coor.append(coor)
        # Arrow
        elif pxl > 1500:
            arrow_coord = coor

    # Cleaning of objects 
    for pt in avg_coor:
        dist = distance(pt, arrow_coord)
        if dist < 70:
            avg_coor.remove(pt)
    to_remove = []
    for i in range(len(avg_coor)):
        for j in range(i, len(avg_coor)):
            dist = distance(avg_coor[i], avg_coor[j])
            if dist > 1e-5 and dist < 50:
                to_remove.append(set(i,j))
    print(to_remove)
            

    return avg_coor, arrow_coord


def crop_digit(frame, digit_pos, size=20):
    """
    Crop the frame out of a specific centroid
    Params:
        digit : (x, y) digit centroid position
        frame : frame to crop
    Return:
        crop : the cropped image
    """
    cropped_img = frame[int(digit_pos[1] - size): int(digit_pos[1] + size),
                        int(digit_pos[0] - size): int(digit_pos[0] + size)]
    return cropped_img

def distance(pt1, pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
