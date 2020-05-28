import numpy as np
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import manifold
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix



class Mlle:

    def __init__(self):
        self.neighbors = 25
        self.components = 2
        self.solver = 'arpack'
        self.model = None

    def load_model(self, filename):
        try:
            self.model = load(filename)
        except:
            print('Could not load model {}'.format(filename))

    def reduce_data(self, x_train, x_test):
        """
        Reduce the data dimentionality using Modified Locally Linear Embedding
        """
        x = np.append(x_train, x_test, axis=0)
        idx = range(x_train.shape[0])
        reverse_idx = range(x_train.shape[0], x_train.shape[0] + x_test.shape[0])

        self.model = manifold.LocallyLinearEmbedding(
            n_neighbors=self.neighbors,
            n_components=self.components,
            method='modified',
            eigen_solver=self.solver,
            n_jobs=-1
        )
        
        x_red = self.model.fit_transform(x)
        
        dump(self.model, 'mlle.h5')

        x_m_train = x_red[idx, :]
        x_m_test = x_red[reverse_idx, :]

        return x_m_train, x_m_test

    def knn(self, x_train, y_train, x_test, y_test):
        """
        """
        acc = []
        for n in range(1, 5):
            self.classifier = KNeighborsClassifier(n_neighbors=n)
            self.classifier.fit(x_train, y_train.ravel())
            acc_knn = self.classifier.score(x_test, y_test.ravel())
            acc.append(acc_knn)
            print('KNN {} acc : {:.02f}'.format(n, acc_knn))

        return acc

    def train(self, classes=[2, 3, 7]):
        """
        Train Modified Locally Linear Embedding with specific classes of digits
        """
        x, y = load_digits(return_X_y=True)
        idx = np.empty((0, 1), dtype='int8')
        for i in classes:
            n = np.argwhere(y == i)
            idx = np.vstack((idx, n))
        x = x[idx]
        x = x.reshape((x.shape[0], -1))
        y = y[idx]
        y = y.astype(np.int)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=120*len(classes),
            test_size=30*len(classes)
        ) 

        x_m_train, x_m_test = self.reduce_data(x_train, x_test)

        self.knn(x_m_train, y_train, x_m_test, y_test)
        

    def predict(self, digit_frame):
        """
        Predict the digit from a frame, but with a resize to fit our Net
        """
        if self.model is None:
            raise NameError('model has not been trained')
        resized = cv2.resize(digit_frame, (8, 8), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = gray.reshape(-1, 64, 1)
        ### We need to invert the image as the background is represented as 0 value in MNIST
        resized = cv2.bitwise_not(resized)
        resized = resized / 255
        print(resized.shape)
        reduced = self.model.transform(resized)
        prediction = self.classifier.predict(reduced)

        return prediction
