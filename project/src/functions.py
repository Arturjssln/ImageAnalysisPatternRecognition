from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as filt

def process(frame):
    # do processing HERE
    return frame


def threshold(img, th1 = None, th2 = None):
    """Threshold function"""
    out = np.ones(np.shape(img))
    if th1 is not None:
        out[np.where(img < th1)] = 0
    if th2 is not None:
        out[np.where(img > th2)] = 0
    return out

def check(list, val):
    """Function to check if a neighbor has label, if yes, it will return the label"""
    if val > min(list) and min(list) > 0:
        return min(list)
    return False


def labeling(img, label):
    """Return an image with element corresponding to the given label"""
    out = np.zeros(np.shape(img), dtype = np.uint)
    out[np.where(img == label)] = 1
    return out


def coor_object(object):
    nn_zero_col = np.any(object, 0)
    nn_zero_row = np.any(object, 1)
    col = np.linspace(0, len(object[0]) - 1, num=len(object[0]), dtype=int)
    row = np.linspace(0, len(object) - 1, num=len(object), dtype=int)
    x = np.mean(col[nn_zero_col])
    y = np.mean(row[nn_zero_row])
    return [x, y]

def extract_object(image, label):
    out = np.zeros(np.shape(image))
    out[np.where(image == label)] = 1
    # get amount of pixels forming the object
    pxl = sum(sum(out))
    # get avg coordinates of object
    coor = coor_object(out)
    return out, pxl, coor


def find_objects(frame):
    # reshape : remove black lines on the side (out of table)
    image = np.delete(frame, range(100), axis=1)
    image = np.delete(image, range(500, len(frame[0])), axis=1)
    # from RGB colors to gray scale
    image = rgb2gray(image)
    # threshold on image to separate background from foreground
    # We are using an adaptive threshold that find the best threshold depending of the illumination
    t = filt.threshold_otsu(image)
    image_thres = threshold(image, t)

    fig, axes = plt.subplots(1, 3, figsize=(6, 6))
    axes[0].imshow(image)
    axes[1].imshow(image_thres)

    # do processing HERE
    new_label = 1
    im_h = len(image_thres)
    im_w = len(image_thres[0])

    label = np.zeros((im_h,im_w), dtype=np.uint)

    # Iterating on each pixels of the image
    nb_neighbors = 10
    for i in range(1, im_h - nb_neighbors):
        for j in range(1, im_w - nb_neighbors):
            # Labeling each pixels of the foreground with same value as its neighbor
            if image_thres[i, j] == 0:
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


    axes[2].imshow(label)

    label_unique = np.unique(label)
    # Background is not an object so we subtract 1
    nb_objects = len(label_unique) - 1

    print('Frame 0 : There are {:d} objects.'.format(nb_objects),'labels are', label_unique)

    # plotting only object of a certain label
    fig2, ax = plt.subplots(int(nb_objects / 5)+1, 5, figsize=(14, 8))
    nb_pixels = []
    avg_coor = []
    for label_choice in range(1,nb_objects+1):
        img, pxl, coor = extract_object(label, label_unique[label_choice])
        ax[int(label_choice / 5)][label_choice % 5].imshow(img)
        ax[int(label_choice / 5)][label_choice % 5].set_title('%i pixels'%pxl)
        ax[int(label_choice / 5)][label_choice % 5].scatter(coor[0], coor[1], s = 10, color='r', marker = 'o')
        nb_pixels.append(pxl)
        avg_coor.append(coor)
    plt.pause(100)

    #return frame


