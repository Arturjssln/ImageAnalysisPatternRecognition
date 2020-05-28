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
        x_train = np.concatenate((x_augmented, x_train))
        y_train = np.concatenate((y_augmented, y_train))
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
<<<<<<< Updated upstream
=======
        #train_x, train_y = self.data_augmentation(train_x, train_y, 50000)
        #test_x, test_y = self.data_augmentation(test_x, test_y, 5000)
>>>>>>> Stashed changes

        train_x, train_y = self.data_augmentation(train_x, train_y, 5000)
        test_x, test_y = self.data_augmentation(test_x, test_y, 5000)

        train_y_one_hot = to_categorical(train_y)
        test_y_one_hot = to_categorical(test_y)

        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
        self.model.add(Activation('relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(9))
        self.model.add(Activation('softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

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
<<<<<<< Updated upstream
        resized = gray.reshape(-1, 28, 28, 1)  
=======

        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        resized = gray.reshape(-1, 28, 28, 1)
>>>>>>> Stashed changes
        ### We need to invert the image as the background is represented as 0 value in MNIST
        resized = cv2.bitwise_not(resized)      
        resized = resized / 255
<<<<<<< Updated upstream
        prediction = self.model.predict(resized)
        prediction = prediction[0]
        return prediction
=======

        preds = np.empty((0, 10), dtype='float32')
        for angle in range(0, 360, 45):
            im = self.rotate_image(resized.reshape(28, 28, 1), angle)
            plt.figure()
            plt.imshow(im)
            plt.show(block=True)
            im = im.reshape(-1, 28, 28, 1)
            p = self.model.predict(im)[0]
            print(p.shape)
            preds = np.vstack((preds, p))

        preds = preds.ravel()

        # prediction = self.model.predict(resized)
        # prediction = prediction[0]
        print(np.argmax(preds) % 10)
        return preds


    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
>>>>>>> Stashed changes
