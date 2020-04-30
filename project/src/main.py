import argparse
import cv2

parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--input',
                    type = str, default = '../data/robot_parcours_1.avi',
                    help = 'input path video')

parser.add_argument('--output',
                    type = str, default = '../results/robot_parcours_1.avi',
                    help = 'output result path video')

args = parser.parse_args()


def process(frame):
    # do processing HERE
    return frame


def main():
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
            out.write(frame)
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
