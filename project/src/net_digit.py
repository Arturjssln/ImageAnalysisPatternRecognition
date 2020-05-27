import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
import cv2

class Net:

    def __init__(self):
        self.model = None

    def load_model(self, filename='model.h5'):
        try:
            self.model = load_model(filename)
        except:
            print('could not load model')

    def train(self):
        """
        """
        (train_x,train_y), (test_x,test_y) = mnist.load_data()

        train_x = train_x.reshape(-1, 28, 28, 1)
        test_x = test_x.reshape(-1, 28, 28, 1)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255
        test_x = test_x / 255

        train_y_one_hot = to_categorical(train_y)
        test_y_one_hot = to_categorical(test_y)

        self.model = Sequential()

        self.model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(64, (3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

        self.model.fit(train_x, train_y_one_hot, batch_size=64, epochs=5)

        test_loss, test_acc = self.model.evaluate(test_x, test_y_one_hot)
        print('Test loss', test_loss)
        print('Test accuracy', test_acc)

        predictions = self.model.predict(test_x)
        print(np.argmax(np.round(predictions[0])))

        self.model.save("model.h5")

    def predict(self, digit_frame):
        """
        Predict the digit from a frame, but with a resize to fit our Net
        """
        if self.model is None:
            raise NameError('model has not been trained')
        resized = cv2.resize(digit_frame, (28, 28), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = gray.reshape(-1, 28, 28, 1)  
        ### We need to invert the image as the background is represented as 0 value in MNIST
        resized = cv2.bitwise_not(resized)      
        resized = resized / 255
        prediction = self.model.predict(resized)
        prediction = prediction[0]
        return prediction