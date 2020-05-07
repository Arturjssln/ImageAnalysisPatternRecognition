"""
Example of how to use ditig handling
"""
import argparse
import cv2
import numpy as np
from net import Net
from helper import find_contour, find_descriptor, convert_contour
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import Augmentor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib

def augment_image(positions, frame):
    for op, nb in zip(positions, range(len(positions))):
        crop = crop_digit(op, frame)
        cv2.imwrite('../data/{}/main.jpg'.format(nb), crop)
        p = Augmentor.Pipeline("../data/{}".format(nb)) 
        p.flip_left_right(0.5) 
        p.black_and_white(0.1) 
        p.rotate(0.3, 25, 25) 
        p.skew(0.4, 0.5) 
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
        p.sample(50) 
    

def train_descriptors():
    descriptor_list = [] # should reach 4 depth with each is a list of the descriptors
    opencv_version, opencv_version_minor, _ = cv2.__version__.split(".")
    
    plt.figure()
    for i in range(4):
        filenames = [img for img in glob.glob("../data/{}/output/*.jpg".format(i))]
        filenames.sort() # ADD THIS LINE
        images = [cv2.imread(img) for img in filenames]

        descript_ = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
            contour = find_contour(img, opencv_version)
            contour_complex = convert_contour(contour)
            descriptor = find_descriptor(contour_complex)
            start_descriptor = 0
            descript_.append(abs(descriptor[start_descriptor:start_descriptor+4]))
        descript_ = np.asarray(descript_)
        descriptor_list.append(descript_)
        plt.scatter(descript_[:, 0], descript_[:, 1], label=i)

    plt.legend()
    plt.show(block = True)

    ### SVC 
    x_train = np.concatenate((descriptor_list[0], descriptor_list[1], descriptor_list[2], descriptor_list[3]), axis=0)
    y_train = np.concatenate((np.zeros(len(descriptor_list[0]), dtype=int), np.ones(len(descriptor_list[1]), dtype=int),np.ones(len(descriptor_list[2]), dtype=int)*2,np.ones(len(descriptor_list[3]),dtype=int)*3), axis=0)

        # Create the parameter grid based on the results of random search 
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    S = GridSearchCV(SVC(), params_grid, cv=5, verbose=10)
    S.fit(x_train, y_train)
    print('Best score for training data: {}'.format(S.best_score_))
    print('Best C: {}'.format(S.best_estimator_.C))
    print('Best Kernel: {}'.format(S.best_estimator_.kernel))
    print('Best Gamma: {}'.format(S.best_estimator_.gamma))
    final_model = S.best_estimator_
    joblib.dump(S.best_estimator_, 'model_op.pkl', compress=1)
    print("Training set score for SVM: %f" % final_model.score(x_train, y_train))

def process(frame):
    digit_positions = np.array([[295, 104], [161, 94], [370, 205], [198, 285]])
    augment_image(digit_positions, frame)
    train_descriptors()
    return frame

