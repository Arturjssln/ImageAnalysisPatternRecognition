import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

class Net:

    def __init__(self):
        """
        Init function
        """
        self.model = None

    def load_model(self, filename):
        """
        Load Model function
        """
        try:
            self.model = load_model(filename)
        except:
            print('Could not load model {}'.format(filename))

    def remove(self, digit, x, y):
        """
        Remove a digit from MNIST
        """
        idx = (y != digit).nonzero()
        return x[idx], y[idx]


    def data_augmentation(self, x_train, y_train, augment_size=5000):
        """
        Aumgent data
        """
        image_generator = ImageDataGenerator(
            rotation_range=360,
            zoom_range = 0,
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=False)
        # fit data for zca whitening
        image_generator.fit(x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(x_train.shape[0], size=augment_size)
        x_augmented = x_train[randidx].copy()
        y_augmented = y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size), batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        x_train = np.concatenate((x_train, x_augmented))
        y_train = np.concatenate((y_train, y_augmented))
        return x_train, y_train

    def train(self):
        """
        """
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        train_x, train_y = self.remove(9, train_x, train_y)
        test_x, test_y = self.remove(9, test_x, test_y)

        train_x = train_x.reshape(-1, 28, 28, 1)
        test_x = test_x.reshape(-1, 28, 28, 1)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255
        test_x = test_x / 255
        train_x, train_y = self.data_augmentation(train_x, train_y, 50000)
        test_x, test_y = self.data_augmentation(test_x, test_y, 5000)

        train_y_one_hot = to_categorical(train_y)
        test_y_one_hot = to_categorical(test_y)

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(9))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        #TODO SVEN change the line below
        self.model.fit(train_x[:10000], train_y_one_hot[:10000], batch_size=64, epochs=15)

        test_loss, test_acc = self.model.evaluate(test_x, test_y_one_hot)
        print('Test loss', test_loss)
        print('Test accuracy', test_acc)
        predictions = self.model.predict(test_x)
        self.model.save("model.h5")

    def predict(self, digit_frame, rotate_model):
        """
        Predict the digit from a frame
        """
        if self.model is None:
            raise NameError('model has not been trained')
        resized = cv2.resize(digit_frame, (28, 28), interpolation = cv2.INTER_CUBIC)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = resized.reshape(-1, 28, 28, 1)
        ### We need to invert the image as the background is represented as 0 value in MNIST
        resized = cv2.bitwise_not(resized)
        resized = resized / 255
        if rotate_model is not None:
            resized = rotate_model.correct_rotation(resized)
            resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_CUBIC).reshape(-1, 28, 28, 1)
        prediction = self.model.predict(resized)
        prediction = prediction[0]
        return prediction
