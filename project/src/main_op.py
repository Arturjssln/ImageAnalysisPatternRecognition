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
import glob
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib




parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--input',
                    type = str, default = '../data/robot_parcours_1.avi',
                    help = 'input path video')

parser.add_argument('--output',
                    type = str, default = '../results/robot_parcours_1.avi',
                    help = 'output result path video')

args = parser.parse_args()


def crop_digit(digit, frame, HALF_WIDTH = 20):
    """
    Crop the frame out of a specific centroid
    Params:
        digit : (x, y) digit centroid position 
        frame : frame to crop
    Return:
        crop : the cropped image
    """
    crop_img = frame[(digit[1] - HALF_WIDTH) : (digit[1] + HALF_WIDTH), (digit[0] - HALF_WIDTH) : (digit[0] + HALF_WIDTH) ]
    return crop_img

def predict_digit(img):
    """
    Predict the digit from the cropped image with the help of Net module
    Params:
        img: Image we want to predict
    Return:
        prediction as integer.
    """
    global model
    prediction = model.predict(img)
    return np.argmax(prediction)

def process_digits(digit_positions, frame):
    """
    Process the list of digits positions and give the prediction for each. 
    Params:
        digit_positions: list of digits position (x, y)
        frame: the frame to use
    """
    #processed digits
    for digit in digit_positions:
        break
        digit_frame = crop_digit(digit, frame)
        predicted = predict_digit(digit_frame)
        cv2.imshow("{}".format(predicted), digit_frame)
        cv2.waitKey(0)

def augment_image():
    for i in range(5):
        p = Augmentor.Pipeline("../data/{}".format(i)) 
        p.flip_left_right(0.5) 
        p.black_and_white(0.1) 
        p.rotate(0.3, 25, 25) 
        p.skew(0.4, 0.5) 
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
        p.sample(50) 
    

def train_descriptors():
    descriptor_list = [] # should reach 5 depth with each is a list of the descriptors
    opencv_version, _, _ = cv2.__version__.split(".")
    
    plt.figure()
    for i in range(5):
        filenames = [img for img in glob.glob("../data/{}/output/*.jpg".format(i))]
        filenames.sort() # ADD THIS LINE
        images = [cv2.imread(img) for img in filenames]

        descript_ = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
            contour = find_contour(img, opencv_version)
            contour_complex = convert_contour(contour)
            descriptor = find_descriptor(contour_complex)
            start_descriptor = 0
            descript_.append(abs(descriptor[start_descriptor:start_descriptor+4]))
        descript_ = np.asarray(descript_)
        descriptor_list.append(descript_)

    ### SVC 
    x_train = np.concatenate((descriptor_list[0], descriptor_list[1], descriptor_list[2], descriptor_list[3], descriptor_list[4]), axis=0)
    y_train = np.concatenate((np.zeros(len(descriptor_list[0]), dtype=int), np.ones(len(descriptor_list[1]), dtype=int), np.ones(len(descriptor_list[2]), dtype=int)*2, np.ones(len(descriptor_list[3]),dtype=int)*3, np.ones(len(descriptor_list[4]),dtype=int)*4), axis=0)

    # Create the parameter grid based on the results of random search 
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    S = GridSearchCV(SVC(), params_grid, cv=2, verbose=10)
    S.fit(x_train, y_train)
    print('Best score for training data: {}'.format(S.best_score_))
    print('Best C: {}'.format(S.best_estimator_.C))
    print('Best Kernel: {}'.format(S.best_estimator_.kernel))
    print('Best Gamma: {}'.format(S.best_estimator_.gamma))
    final_model = S.best_estimator_
    joblib.dump(S.best_estimator_, 'model_op.pkl', compress=1)
    print("Training set score for SVM: %f" % final_model.score(x_train, y_train))

def process(frame, augment):
    digit_positions = np.array([[295, 104], [161, 94], [370, 205], [198, 285]])
    if augment :
        augment_image()
    train_descriptors()
    return frame


model = Net()

def main():
    print('Importing file')
    AUGMENT_IMAGES = True
    
    # Playing video from file:
    cap = cv2.VideoCapture(filename = args.input)
    # Define the codec and create VideoWriter object.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(filename = args.output, fourcc = cv2.VideoWriter_fourcc('M','J','P','G'), fps = 2, frameSize = (frame_width, frame_height))

    currentFrame = 0
    while(cap.isOpened()):
        # Capture frame by frame
        ret, frame = cap.read()
        if ret:
            print('Processing frame #{}'.format(currentFrame))
            processed_frame = process(frame, AUGMENT_IMAGES)
            out.write(processed_frame)
            break ######################## TO REMOVE THIS LINE ###########################
        else:
            break

        # To stop duplicate images
        currentFrame += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('Done...')


if __name__ == '__main__':
    main()
