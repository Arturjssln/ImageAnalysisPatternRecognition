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

def augment_image():
    for i in range(5):
        p = Augmentor.Pipeline("../data/{}".format(i)) 
        p.flip_left_right(0.5) 
        p.rotate(0.3, 25, 25) 
        p.skew(0.4, 0.5) 
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
        p.sample(100) 
    

def train_model():

    x_data = []
    y_data = []
    for i in range(5):
        filenames = [img for img in glob.glob("../data/{}/output/*".format(i))]
        filenames.sort() # ADD THIS LINE
        images = [cv2.imread(img) for img in filenames]

        descript_ = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
            x_data.append(img)
            y_data.append(i)
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    global model
    model.train(x_data, y_data)



def process(frame, augment):

    if augment :
        augment_image()
    train_model()
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
