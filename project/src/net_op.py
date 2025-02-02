import keras
from keras.datasets import mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import glob
import Augmentor

class NetOp:

    def __init__(self, augment):
        self.model = None
        self.augment = augment
        if self.augment:
            self.augment_image()

    def load_model(self, filename):
        try:
            self.model = load_model(filename)
        except:
            print('could not load model {}'.format(filename))

    def augment_image(self):
        for i in range(5):
            p = Augmentor.Pipeline("../data/{}".format(i)) 
            p.flip_left_right(0.5) 
            p.rotate(0.3, 25, 25) 
            p.skew(0.4, 0.5) 
            p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
            p.sample(1000) 

    def train(self):
        """
        """
        x_data = []
        y_data = []
        for i in range(5):
            filenames = [img for img in glob.glob("../data/{}/output/*".format(i))]
            filenames.sort()
            images = [cv2.imread(img) for img in filenames]

            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

                x_data.append(img)
                y_data.append(i)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3)
        train_x = train_x.reshape(-1, 28, 28, 1)
        test_x = test_x.reshape(-1, 28, 28, 1)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255
        test_x = test_x / 255

        train_y_one_hot = to_categorical(train_y)
        test_y_one_hot = to_categorical(test_y)

        self.model = Sequential()

        self.model.add(Conv2D(512, (3,3), input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(128, (3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(256))

        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        self.model.summary()
        self.model.fit(train_x, train_y_one_hot, batch_size=64, epochs=6)

        test_loss, test_acc = self.model.evaluate(test_x, test_y_one_hot)
        print('Test loss', test_loss)
        print('Test accuracy', test_acc)

        self.model.save("operators.h5")

    def predict(self, op_img):
        """
        Predict the digit from a frame, but with a resize to fit our Net
        """
        img = cv2.cvtColor(op_img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        resized = img.reshape(-1, 28, 28, 1)
        resized = resized / 255
        prediction = self.model.predict(resized)
        prediction = np.argmax(prediction)
        #print(prediction)
        return prediction