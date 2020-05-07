"""
main program
"""
import argparse
import numpy as np
import cv2
from net import Net
from utils import find_objects, coor_object, crop_digit

parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--input',
                    type=str, default='../data/robot_parcours_1.avi',
                    help='input path video')

parser.add_argument('--output',
                    type=str, default='../results/robot_parcours_1.avi',
                    help='output result path video')

args = parser.parse_args()

class Calculator:
    """
    Class that contains all functions necessary to compute the calculs
    """
    def __init__(self, input_path, output_path):
        """
        Initialization of the class
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.out = None
        self.object_position = None
        self.arrow_position = None
        self.arrow_color = None
        self.closest_pos = None
        self.equation = ''
        self.proximity_threshold = 20
        self.model =  Net()
        self.model.load_model()
        self.initial_frame = None

    def __enter__(self):
        """
        Importing the input video and creating the output video
        """
        print('Importing file...')
        # Using video from file:
        self.cap = cv2.VideoCapture(filename=self.input_path)
        # Define the codec and create VideoWriter object.
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        self.out = cv2.VideoWriter(filename=self.output_path, fourcc=cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps=2, frameSize=(frame_width, frame_height))

    def __exit__(self, type_, value_, traceback_):
        """
        Releasing the input and output video
        """
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print('Output video saved...')
        self.cap = None
        self.out = None

    def process(self):
        """
        Process frames one by one to analyze
        """
        if self.cap is None or self.out is None:
            print("ERROR : Use this function inside a with statement")
            return
        current_frame = 0
        while self.cap.isOpened():
            # Capture frame by frame
            ret, frame = self.cap.read()
            if ret:
                print('Processing frame #{}'.format(current_frame))
                if current_frame == 0:
                    self.object_position, self.arrow_position, self.arrow_color = \
                        find_objects(frame)
                    self.initial_frame = frame.copy()
                else:
                    self.find_arrow(frame)
                    self.compute_closest_object()
                    self.add_to_equation()
                self.out.write(self.frame_display(frame))
            else:
                break
            current_frame += 1

    def find_arrow(self, frame):
        """
        Find the position of the arrow using the known color of the arrow
        """
        lower_red = np.array(self.arrow_color)*255 - np.array([70, 70, 70])
        upper_red = np.array(self.arrow_color)*255 + np.array([70, 70, 70])
        mask = cv2.inRange(frame, lower_red, upper_red)
        closing = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        self.arrow_position = coor_object(closing)

    def compute_closest_object(self):
        """
        Find the closest object
        """
        min_dist = np.inf
        for pos in self.object_position:
            dist = (pos[0]-self.arrow_position[0])**2 + \
                (pos[1]-self.arrow_position[1])**2
            if dist < min_dist:
                self.closest_pos = pos
                min_dist = dist

    def add_to_equation(self):
        """
        Add new object to equation if it is close enough
        """
        if (self.closest_pos[0]-self.arrow_position[0])**2 + \
            (self.closest_pos[1]-self.arrow_position[1])**2 < \
            self.proximity_threshold**2:
            predicted = str(self.predict_object())
            # If the equation is empty or if it is a new object, we add it
            if len(self.equation) == 0 or predicted != self.equation[-2]:
                self.equation += predicted + ' '

    def predict_object(self):
        """
        Predict object at closest position
        """
        return self.predict_digit(self.closest_pos)

    def predict_digit(self, digit_pos):
        """
        Predict the digit at the position digit_pos
        Params:
            digit_pos: Position of the center of the digit
        Return:
            prediction as integer.
        """
        digit_frame = crop_digit(self.initial_frame, digit_pos)
        prediction = self.model.predict(digit_frame)
        return np.argmax(prediction)

    def frame_display(self, frame):
        """
        Prepare frame for video
        """
        out = self.display_objects(frame)
        out = self.display_equation(frame)
        return out

    def display_objects(self, frame):
        """
        Display object position on frame
        """
        size = np.array([20, 20])
        # Draw digit and sign cases
        object_position = [np.array(elt) for elt in self.object_position]
        for pos in object_position:
            top_left = pos - size
            bottom_right = pos + size
            frame = cv2.rectangle(
                frame, (int(top_left[0]), int(top_left[1])),
                (int(bottom_right[0]), int(bottom_right[1])),
                (255, 0, 0), thickness=2)
        # Draw arrow case
        top_left = np.array(self.arrow_position) - 2*size
        bottom_right = np.array(self.arrow_position) + 2*size
        frame = cv2.rectangle(
            frame, (int(top_left[0]), int(top_left[1])),
            (int(bottom_right[0]), int(bottom_right[1])),
            (0, 0, 255), thickness=2)
        return frame

    def display_equation(self, frame):
        """
        Display equation on frame
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(self.equation, font, 1, 2)[0]
        text_x = (frame.shape[1] - textsize[0]) // 2
        text_y = (frame.shape[0] + textsize[1]) * 3 // 4
        frame = cv2.putText(frame, self.equation,
                            (text_x, text_y), font, 1, (255, 0, 0), 2)
        return frame

def main():
    """
    main function
    """
    calculator = Calculator(args.input, args.output)
    with calculator:
        calculator.process()


if __name__ == '__main__':
    main()
