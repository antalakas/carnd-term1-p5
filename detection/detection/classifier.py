import time
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import pickle

import os
import os.path

from .features import Features


class Classifier(object):
    def __init__(self, base_directory, small=True):
        self.features = Features()

        self.cars = []
        self.car_features = []

        self.notcars = []
        self.notcar_features = []

        self.X_train = []
        self.X_test = []

        self.y_train = []
        self.y_test = []

        self.svc = None

        self.extension = ''

        if small:
            self.path = os.path.join(base_directory, 'datasets\small')
            self.extension = '.jpeg'
        else:
            self.path = os.path.join(base_directory, 'datasets\large')
            self.extension = '.png'

    def load_data(self):
        path_cars = os.path.join(self.path, 'vehicles\**\*' + self.extension)
        self.cars = glob.glob(path_cars)
        self.notcars = glob.glob(os.path.join(self.path, 'non-vehicles\**\*' + self.extension))

        print('Cars', len(self.cars))
        print ('Non-Cars', len(self.notcars))

    def extract_features(self, color_space='RGB',
                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

        t = time.time()
        features_path = os.path.join(self.path, 'features.p')

        if os.path.isfile(features_path):
            print('Loading HOG features from file...')
            handle = open(features_path, 'rb')
            dist_pickle = pickle.load(handle)
            self.car_features = dist_pickle["car_features"]
            self.notcar_features = dist_pickle["notcar_features"]
        else:
            print('Extracting HOG features...')
            self.car_features = self.features.extract_features(self.cars, color_space=color_space,
                                            orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel)
            self.notcar_features = self.features.extract_features(self.notcars, color_space=color_space,
                                               orient=orient, pix_per_cell=pix_per_cell,
                                               cell_per_block=cell_per_block,
                                               hog_channel=hog_channel)

            dist_pickle = {}
            dist_pickle["car_features"] = self.car_features
            dist_pickle["notcar_features"] = self.notcar_features

            pickle_file = os.path.join(self.path, 'features.p')
            pickle.dump(dist_pickle, open(pickle_file, "wb"))

        t2 = time.time()
        print(round(t2 - t, 2), 'Time to extract HOG features (sec)...')

        X = np.vstack((self.car_features, self.notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(self.car_features)), np.zeros(len(self.notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(self.X_train[0]))

    def train(self):
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(self.X_train, self.y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(self.X_test, self.y_test), 4))
