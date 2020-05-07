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


def train_descriptors():
    descriptor_list = []  # should reach 4 depth with each is a list of the descriptors
    opencv_version, opencv_version_minor, _ = cv2.__version__.split(".")

    plt.figure()
    for i in range(4):
        filenames = [img for img in glob.glob(
            "../data/{}/output/*.jpg".format(i))]
        filenames.sort()  # ADD THIS LINE
        images = [cv2.imread(img) for img in filenames]

        descript_ = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            contour = find_contour(img, opencv_version)
            contour_complex = convert_contour(contour)
            descriptor = find_descriptor(contour_complex)
            start_descriptor = 0
            descript_.append(
                abs(descriptor[start_descriptor:start_descriptor+4]))
        descript_ = np.asarray(descript_)
        descriptor_list.append(descript_)
        plt.scatter(descript_[:, 0], descript_[:, 1], label=i)

    plt.legend()
    plt.show(block=True)

    ### SVC
    x_train = np.concatenate(
        (descriptor_list[0], descriptor_list[1], descriptor_list[2], descriptor_list[3]), axis=0)
    y_train = np.concatenate((np.zeros(len(descriptor_list[0]), dtype=int), np.ones(len(descriptor_list[1]), dtype=int), np.ones(
        len(descriptor_list[2]), dtype=int)*2, np.ones(len(descriptor_list[3]), dtype=int)*3), axis=0)

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
    print("Training set score for SVM: %f" %
          final_model.score(x_train, y_train))
