import keras
from keras.datasets import mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, apply_affine_transform
from keras.models import Sequential, load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Net:

    def __init__(self):
        pass

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

        print(train_x.shape, test_x.shape)
        train_x, train_y = self.augment_data(train_x[:1000], train_y[:1000])
        test_x, test_y = self.augment_data(test_x[:1000], test_y[:1000])
        print(train_x.shape, test_x.shape)

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

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

        self.model.fit(train_x, train_y_one_hot, batch_size=64, epochs=10)

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
        print(digit_frame.shape)
        resized = cv2.resize(digit_frame, (28, 28), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = gray.reshape(-1, 28, 28, 1)  
        ### We need to invert the image as the background is represented as 0 value in MNIST
        resized = cv2.bitwise_not(resized) 
        resized = resized / 255
        prediction = self.model.predict(resized)
        prediction = prediction[0]
        print(prediction)
        return prediction

    def data_augmentation(self, x_train, y_train, augment_size=50000):
        image_generator = ImageDataGenerator(rotation_range=90)
        image_generator.fit(x_train, augment=True)
        randidx = np.random.randint(x_train.shape[0], size=augment_size)
        x_augmented = x_train[randidx].copy()
        y_augmented = y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, batch_size=augment_size, shuffle=False)
        return np.concatenate((x_train, x_augmented)), np.concatenate((y_train, y_augmented))

    def augment_data(self, train_x, train_y):
        augmented_image = []
        augmented_image_labels = []

        for num in range(0, train_x.shape[0]):
            for angle in np.arange(0, 360, 10):
                augmented_image.append(apply_affine_transform(train_x[num], theta=angle, fill_mode='constant', row_axis=0, col_axis=1, channel_axis=2))
                augmented_image_labels.append(train_y[num])

        return np.array(augmented_image), np.array(augmented_image_labels)
