import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--input',
                    type=str, default='../data/robot_parcours_1.avi',
                    help='input path video')

parser.add_argument('--output',
                    type=str, default='../results/robot_parcours_1.avi',
                    help='output result path video')

args = parser.parse_args()


def find_object(frame):
    """
    Find object and arrow position in frame and return it
    """
    return frame, frame

def display_objects(frame, object_position, arrow_position):
    """
    Display object position on frame
    """
    size = np.array([20, 20])
    # Draw digit and sign cases
    object_position = [np.array(elt) for elt in object_position]
    for pos in object_position:
        top_left = pos - size
        bottom_right = pos + size
        frame = cv2.rectangle(
            frame, (int(top_left[0]), int(top_left[1])), 
            (int(bottom_right[0]), int(bottom_right[1])), 
            (255, 0, 0), thickness=2)
    # Draw arrow case
    top_left = np.array(arrow_position) - 2*size
    bottom_right = np.array(arrow_position) + 2*size
    frame = cv2.rectangle(
        frame, (int(top_left[0]), int(top_left[1])),
        (int(bottom_right[0]), int(bottom_right[1])),
        (0, 0, 255), thickness=2)
    return frame

def process(frame):
    """
    Processing image
    """
    object_position, arrow_position = find_object(frame)
    object_position = np.asarray([[283, 233], [468, 236], [
        487, 127], [295, 104], [161, 94], [370, 205], [198, 285]])
    arrow_position = [468, 300]
    processed_frame = display_objects(frame, object_position, arrow_position)
    return processed_frame


def main():
    """
    main function
    """
    print('Importing file...')
    # Playing video from file:
    cap = cv2.VideoCapture(filename=args.input)
    # Define the codec and create VideoWriter object.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(filename=args.output, fourcc=cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps=2, frameSize=(frame_width, frame_height))
    out2 = cv2.VideoWriter(filename='../results/robot_parcours_2.avi', fourcc=cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), fps=2, frameSize=(frame_width, frame_height))  
    try:
        current_frame = 0
        while cap.isOpened():
            # Capture frame by frame
            ret, frame = cap.read()
            if ret:
                print('Processing frame #{}'.format(current_frame))
                processed_frame = process(frame)
                out.write(processed_frame)
            else:
                break

            # To stop duplicate images
            current_frame += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print('Done...')


if __name__ == '__main__':
    main()
