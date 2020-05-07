import heapq
from skimage.color import rgb2gray
import numpy as np
import skimage.filters as filt
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import Augmentor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

def threshold(img, th1 = None, th2 = None):
    """
    Threshold function
    """
    out = np.ones(np.shape(img))
    if th1 is not None:
        out[np.where(img < th1)] = 0
    if th2 is not None:
        out[np.where(img > th2)] = 0
    return out

def check(list, val):
    """
    Function to check if a neighbor has label, if yes, it will return the label
    """
    if val > min(list) and min(list) > 0:
        return min(list)
    return False


def labeling(img, label):
    """
    Return an image with element corresponding to the given label
    """
    out = np.zeros(np.shape(img), dtype = np.uint)
    out[np.where(img == label)] = 1
    return out


def coor_object(object_):
    """
    Return the coordinate of the center of an object
    """
    nn_zero_col = np.any(object_, 0)
    nn_zero_row = np.any(object_, 1)
    col = np.linspace(0, len(object_[0]) - 1, num=len(object_[0]), dtype=int)
    row = np.linspace(0, len(object_) - 1, num=len(object_), dtype=int)
    x = np.mean(col[nn_zero_col])
    y = np.mean(row[nn_zero_row])
    return [x, y]

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


def color_arrow(label, label_unique, frame, label_choice):
    """
    Return the color of an object (only used for arrow)
    """
    img, _, _ = extract_object(label, label_unique[label_choice])
    return [(frame[:, :, i] * img).mean() for i in range(3)] 


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
    t = filt.threshold_otsu(image)
    ones[:, 100:frame.shape[1]-100] = threshold(image, t)
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
                label_neighbors = np.max([label[i + a, j + b] for a in range(- nb_neighbors, nb_neighbors+1)
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
            label_neighbors = [label[i + a, j + b] for a in range(- nb_neighbors2, nb_neighbors2 + 1)
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
    for label_choice in range(1,nb_objects+1):
        _, pxl, coor = extract_object(label, label_unique[label_choice])
        nb_pixels.append(pxl)
        avg_coor.append(coor)
    
    #remove line in middle
    idx = heapq.nlargest(3, range(len(nb_pixels)), key=nb_pixels.__getitem__)[:2]
    arrow_coord = avg_coor[idx[0]]
    arrow_color = color_arrow(label, label_unique, frame, idx[0]+1)
    avg_coor = [i for j, i in enumerate(avg_coor) if j not in idx]
    
    return avg_coor, arrow_coord, arrow_color


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

MIN_CONTOUR_POINT = 20
def find_contour(img, opencv_version):
    """
    Finds and returns the contour of the image
    """
    contour = []
    if int(opencv_version) == 3:
        _, contour, _ = cv2.findContours(
            img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    else:
        contour, _ = cv2.findContours(
            img.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    contour_array = contour[0].reshape(-1, 2)
    # minimum contour size required
    if contour_array.shape[0] < MIN_CONTOUR_POINT:
        contour_array = contour[1].reshape(-1, 2)
    return contour_array


def convert_contour(contour):
    """
    TODO
    """
    contour_complex = np.empty(contour.shape[:-1], dtype=complex)
    contour_complex.real = contour[:, 0]
    contour_complex.imag = contour[:, 1]
    return contour_complex


def find_descriptor(contour):
    """
    Finds and returns the Fourier-Descriptor from the image contour
    """
    return np.fft.fft(contour)

