import argparse
import cv2
from utils import find_objects, coor_object
import numpy as np


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

    def __exit__(self, type, value, traceback):
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
                    self.object_position, self.arrow_position, self.arrow_color = find_objects(frame)
                else:
                    self.find_arrow(frame)
                
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

    def frame_display(self, frame):
        """
        Prepare frame for video
        """
        out = self.display_objects(frame)
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

def main():
    """
    main function
    """
    calculator = Calculator(args.input, args.output)
    with calculator:
        calculator.process()


if __name__ == '__main__':
    main()
