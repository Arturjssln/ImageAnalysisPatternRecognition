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

def remove_borders(image):

    return image
def check(list, val):
    """Function to check if a neighbor has label, if yes, it will return the label"""
    # traverse in the list
    for x_list in list:
        # compare with all the values with val
        if val > x_list and x_list > 0:
            return x_list
    return False


def labeling(img, label):
    """Return an image with element corresponding to the given label"""
    out = np.zeros(np.shape(img), dtype = np.uint)
    out[np.where(img == label)] = 1
    return out

def find_objects(frame):
    plt.imshow(frame)
    #plt.pause(1000)
    # reshape : remove black lines on the side (out of table)
    image = np.delete(frame, range(100), axis=1)
    image = np.delete(image, range(500, len(frame[0])), axis=1)
    plt.imshow(image)
    # plt.pause(1000)
    # from RGB colors to gray scale
    image = rgb2gray(image)
    # threshold on image to separate background from foreground
    # We are using an adaptive threshold that find the best threshold depending of the illumination
    print('maximum gray value:', np.max(image))
    t = filt.threshold_otsu(image)
    print('threshold is', t)
    image_thres = threshold(image, t)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(image)
    axes[1].imshow(image_thres)

    # do processing HERE
    new_label = 1
    im_h = len(image_thres)
    im_w = len(image_thres[0])

    label = np.zeros((im_h,im_w), dtype=np.uint)

    # Iterating on each pixels of the image
    nb_neighbors = 2
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
    label2 = label
    for i in range(1, im_h - 1):
        for j in range(1, im_w - 1):
            label_neighbors = [label2[i - 1, j + 1], label2[i, j + 1], label2[i + 1, j + 1], label2[i - 1, j],
                               label2[i + 1, j], label2[i - 1, j - 1], label2[i, j - 1], label2[i + 1, j - 1]]
            val_min = check(label_neighbors, label2[i, j])
            if check(label_neighbors, label2[i, j]):
                label2[np.where(label2 == label2[i, j])] = val_min

    print('plot the labels')
    axes[2].imshow(label)
    plt.pause(100)

    label_unique = np.unique(label)
    # Background is not an object so we subtract 1
    nb_objects = len(label_unique) - 1

    print('Frame 0 : There are {:d} objects.'.format(nb_objects))
    #return frame
