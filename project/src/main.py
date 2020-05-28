"""
main program
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
import cv2
from sympy import sympify, solve
from net_digit import Net
from net_op import NetOp
from mlle import Mlle
from utils import find_objects, coor_object, crop_digit


parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--input',
                    type=str, default='../data/robot_parcours.avi',
                    help='input path video')

parser.add_argument('--output',
                    type=str, default='../results/robot_parcours.avi',
                    help='output result path video')

parser.add_argument('--train_mlle',
                    action='store_true', default=False,
                    help='Enable the second stage digits training')

parser.add_argument('--train_operators',
                    action='store_true', default=False,
                    help='Enable the operator training')

parser.add_argument('--train_digits',
                    action='store_true', default=False,
                    help='Enable the digits training')

parser.add_argument('--digit_model',
                    type=str, default='model_digit_normal.h5',
                    help='Choice of the model file to use')

parser.add_argument('--operator_model',
                    type=str, default='model_operators.h5',
                    help='Choice of the model file to use')

parser.add_argument('--mlle_model',
                    type=str, default='mlle.h5',
                    help='Choice of the model file to use for the second stage digits')

parser.add_argument('--augment_images',
                    action='store_true', default=False,
                    help='Augment image if selected')

parser.add_argument('--debug',
                    action='store_true', default=False,
                    help='Display debugging mode')

args = parser.parse_args()

class Calculator:
    """
    Class that contains all functions necessary to compute the calculs
    """
    def __init__(self, args):
        """
        Initialization of the class
        """
        self.input_path = args.input
        self.output_path = args.output
        self.debug = args.debug
        self.cap = None
        self.out = None
        self.object_position = None
        self.arrow_position = None
        self.closest_pos = None
        self.equation = ''
        self.proximity_threshold = 30 # pixels
        self.initial_frame = None
        self.current_frame =  None
        self.last_object_pos = None
        self.robot_path = []
        self.model_digit = Net()
        self.model_second_stage = Mlle()
        self.model_operator = NetOp(args.augment_images)
        self.digit_model_path = args.digit_model
        self.operator_model_path = args.operator_model
        self.second_stage_model_path = args.mlle_model

        if args.train_digits:
            print("Training model_digit")
            self.model_digit.train()
        else:
            self.model_digit.load_model(self.digit_model_path)
        
        if args.train_operators:
            print("Training model_operator")
            self.model_operator.train()
        else:
            self.model_operator.load_model(self.operator_model_path)
        
        if args.train_mlle:
            print("Training second_stage")
            self.model_second_stage.train()
        else:
            self.model_second_stage.load_model(self.second_stage_model_path)



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
            raise NameError('Use this function inside a with statement.')
        current_frame = 0
        while self.cap.isOpened():
            # Capture frame by frame
            ret, frame = self.cap.read()
            if ret:
                print('Processing frame #{}'.format(current_frame))
                self.current_frame = frame.copy()
                # Analysis for first frame is more complex than other ones
                if current_frame == 0:
                    self.object_position, self.arrow_position = find_objects(frame)
                    self.initial_frame = frame.copy()
                # Analysis of frame (except fram 0)
                else:
                    self.find_arrow(frame)
                    self.compute_closest_object()
                    self.add_to_equation()
                # Write new frame in the output video
                self.out.write(self.frame_display(frame))
            else:
                break
            current_frame += 1
        # Solve equation and display it at the end.
        self.solve_equation()
        self.out.write(self.frame_display(self.current_frame))

    def find_arrow(self, frame):
        """
        Find the position of the arrow, knowing that it is red
        """
        mask = cv2.inRange(frame, np.array([0, 0, 100]), np.array([100, 100, 255]))
        opening = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        self.arrow_position = coor_object(opening)
        self.robot_path.append(self.arrow_position)


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
            if self.last_object_pos is None or self.last_object_pos != self.closest_pos:
                self.last_object_pos = self.closest_pos
                predicted = str(self.predict_object())
                # If the equation is empty or if it is a new object, we add it
                if len(self.equation) == 0 or predicted != self.equation[-2]:
                    self.equation += predicted + ' '

    def predict_object(self):
        """
        Predict object at closest position
        """
        if len(self.equation) == 0 or not str(self.equation[-2]).isnumeric():
            return self.predict_digit(self.closest_pos)
        return self.predict_operator(self.closest_pos)

    def predict_digit(self, digit_pos):
        """
        Predict the digit at the position digit_pos
        Params:
            digit_pos: Position of the center of the digit
        Return:
            prediction as integer.
        """
        digit_frame = crop_digit(self.initial_frame, digit_pos)
        prediction = self.model_digit.predict(digit_frame)
        prediction = np.argmax(prediction)
        # If number predicted is 9, convert to 6
        if prediction == 9:
            prediction = 6

        return prediction

    def predict_operator(self, operator_pos):
        """
        Predict the operator at the position operator_pos
        Params:
            operator_pos: Position of the center of the operator
        Return:
            prediction as integer.
        """
        operator_frame = crop_digit(self.initial_frame, operator_pos)
        prediction = self.model_operator.predict(operator_frame)
        dictionary = {0: '=', 1: '*', 2: '/', 3: '+', 4: '-'}
        return dictionary[prediction]

    def solve_equation(self):
        """
        Solve equation from string
        """
        # Equation must always end with equal sign
        if str(self.equation[-2]).isnumeric():
            self.equation += '= '
        equation = list(self.equation)
        equation[-2] = ','
        equation += 'x'
        sympy_eq = sympify("Eq(" + "".join(equation).replace(" ", "") + ")")
        result = solve(sympy_eq)
        # Equation must always end with equal sign
        self.equation = self.equation[:-2] + '= '
        self.equation += str(float(result[0]))

    def frame_display(self, frame):
        """
        Prepare frame for video
        """
        out = frame.copy()
        if self.debug:
            out = self.display_objects(out)
        out = self.display_robot_path(out)
        out = self.display_equation(out)
        return out

    def display_robot_path(self, frame):
        """
        Display robot path on frame
        """
        out = frame.copy()
        for i in range(len(self.robot_path)-1):
            start = (int(self.robot_path[i][0]), int(self.robot_path[i][1]))
            end = (int(self.robot_path[i+1][0]), int(self.robot_path[i+1][1]))
            out = cv2.line(out, start, end, color=(255, 255, 255), thickness=1)
        return out

    def display_objects(self, frame):
        """
        Display object position on frame
        """
        out = frame.copy()
        size = np.array([20, 20])
        # Draw digit and sign cases
        object_position = [np.array(elt) for elt in self.object_position]
        for pos in object_position:
            top_left = pos - size
            bottom_right = pos + size
            out = cv2.rectangle(
                out, (int(top_left[0]), int(top_left[1])),
                (int(bottom_right[0]), int(bottom_right[1])),
                (255, 0, 0), thickness=2)
        # Draw arrow case
        top_left = np.array(self.arrow_position) - 2*size
        bottom_right = np.array(self.arrow_position) + 2*size
        out = cv2.rectangle(
            out, (int(top_left[0]), int(top_left[1])),
            (int(bottom_right[0]), int(bottom_right[1])),
            (0, 0, 255), thickness=2)
        return out

    def display_equation(self, frame):
        """
        Display equation on frame
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(self.equation, font, 1, 2)[0]
        text_x = (frame.shape[1] - textsize[0]) // 2
        text_y = (frame.shape[0] + textsize[1]) * 3 // 4
        box_coords = ((text_x-5, text_y+10), (text_x+textsize[0]+5, text_y-textsize[1]-10))
        overlay = frame.copy()
        overlay = cv2.rectangle(overlay, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        overlay = cv2.putText(overlay, self.equation,
                            (text_x, text_y), font, 1, (0, 0, 0), 2)
        frame = cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0)
        return frame

def main():
    """
    main function
    """
    calculator = Calculator(args)
    with calculator:
        calculator.process()


if __name__ == '__main__':
    main()
