from net_digit import Net
import cv2
import numpy as np
import argparse


def rotate_image(self, image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


parser = argparse.ArgumentParser(description='IAPR Special Project.')

parser.add_argument('--file',
                    type=str, default='',
                    help='input path video')

args = parser.parse_args()

model_digit = Net()
model_digit.load_model('.\model_digits_best_4_52.h5')


digit_frame = cv2.imread(args.file, 0)
print(digit_frame.shape)
resized = cv2.resize(digit_frame, (28, 28), interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

resized = gray.reshape(-1, 28, 28, 1)
### We need to invert the image as the background is represented as 0 value in MNIST
resized = cv2.bitwise_not(resized)
resized = resized / 255

preds = np.empty((0, 10), dtype='float32')
for angle in range(0, 360, 45):
    im = rotate_image(resized.reshape(28, 28, 1), angle)
    im = im.reshape(-1, 28, 28, 1)
    p = model.predict(im)[0]
    print(p.shape)
    preds = np.vstack((preds, p))

preds = preds.ravel()

# prediction = self.model.predict(resized)
# prediction = prediction[0]
print(np.argmax(preds) % 10)

