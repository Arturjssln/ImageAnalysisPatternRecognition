"""
Example of how to use ditig handling
"""
import argparse
import cv2
import numpy as np
from net import Net
from pytesseract import *
from PIL import Image
import Augmentor


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
        digit_frame = crop_digit(digit, frame)
        predicted = predict_digit(digit_frame)
        cv2.imshow("{}".format(predicted), digit_frame)
        cv2.waitKey(0)

def process(frame):
    digit_positions = np.array([[282, 233], [468, 236], [487, 127], [295, 104], [161, 94], [370, 205], [198, 285]])
    process_digits(digit_positions, frame)
    return frame

model = Net()

def main():
    #model.train()
    model.load_model()


    print('Importing file')
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
            processed_frame = process(frame)
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
