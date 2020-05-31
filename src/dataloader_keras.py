import math
from scipy import io
import cv2
import numpy as np
import project_path as pp
import sys
import os
import csv
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import keras

class DataBatcher(keras.utils.Sequence):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    def __init__(self, batch_size=150, type=0, shuffle=True):
        self.batch_size = batch_size

        # Assigning .csv file path which contains information about image file paths based on TRAIN / VALIDATE / TEST
        self.csv_file_path = None
        if type == 0:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'train.csv')
        elif type == 1:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'test.csv')
        else:
            self.csv_file_path = os.path.join(pp.generated_folder_path, 'validation.csv')

        # Checking whether the assigned .csv file path exist
        if not os.path.exists(self.csv_file_path):
            print('CSV File Path Does not Exist!')
            print('Quitting...')
            sys.exit(0)

        # Assigning .mat file paths to image means
        self.right_eye_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_right_224.mat')
        self.left_eye_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_left_224.mat')
        self.face_mean_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, 'mean_face_224.mat')

        # Checking whether the assigned .mat file paths exist
        if not (os.path.exists(self.right_eye_mean_file_path) and os.path.exists(self.left_eye_mean_file_path) and os.path.exists(self.face_mean_file_path)):
            print('Mean File Paths (.mat files) Does not Exist!')
            print('Quitting...')
            sys.exit(0)

        # Reading image means
        self.right_eye_mean = io.loadmat(self.right_eye_mean_file_path)['image_mean']
        self.left_eye_mean = io.loadmat(self.left_eye_mean_file_path)['image_mean']
        self.face_mean = io.loadmat(self.face_mean_file_path)['image_mean']

        # Normalizing image means to [0, 1]
        self.right_eye_mean = self.right_eye_mean / 255
        self.left_eye_mean = self.left_eye_mean / 255
        self.face_mean = self.face_mean / 255

        # Reading records from .csv file to get image file locations face grids
        with open(self.csv_file_path) as file:
            csv_reader = csv.reader(file, delimiter=',')
            self.fields = next(csv_reader)
            self.records = []
            for row in csv_reader:
                row[2:6] = [int(item) for item in row[2:6]]
                row[6:] = [float(item) for item in row[6:]]
                self.records.append(row)

        # Calculating hyper-parameters for the current instance of dataset
        self.number_of_instances = len(self.records)
        self.number_of_batches = int(math.ceil(self.number_of_instances / self.batch_size))

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, index):
        record_idxes = self.shuffled_idxes[index * self.batch_size:min(self.number_of_instances, (index + 1) * self.batch_size)]

        right_eye_images = []
        left_eye_images = []
        face_images = []
        face_grids = []
        Y = []

        for record_idx in record_idxes:
            record = self.records[record_idx]

            right_eye_image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleRightEye', record[1] + '.jpg')
            left_eye_image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleLeftEye', record[1] + '.jpg')
            face_image_file_path = os.path.join(pp.gaze_capture_dataset_folder_path, record[0], 'appleFace', record[1] + '.jpg')

            # Checking whether the assigned .mat file paths exist
            if not (os.path.exists(right_eye_image_file_path) and os.path.exists(
                    left_eye_image_file_path) and os.path.exists(face_image_file_path)):
                continue

            # Reading right eye image, swapping color channels, and normalizing it to [0, 1]
            right_eye_image = cv2.cvtColor(cv2.imread(right_eye_image_file_path), cv2.COLOR_BGR2RGB) / 255
            right_eye_image = right_eye_image - self.right_eye_mean
            right_eye_images.append(right_eye_image)

            # Reading left eye image, swapping color channels, and normalizing it to [0, 1]
            left_eye_image = cv2.cvtColor(cv2.imread(left_eye_image_file_path), cv2.COLOR_BGR2RGB) / 255
            left_eye_image = left_eye_image - self.left_eye_mean
            left_eye_images.append(left_eye_image)

            # Reading face image, swapping color channels, and normalizing it to [0, 1]
            face_image = cv2.cvtColor(cv2.imread(face_image_file_path), cv2.COLOR_BGR2RGB) / 255
            face_image = face_image - self.face_mean
            face_images.append(face_image)

            # Reading face grids
            face_grid = np.zeros(shape=(25, 25))
            face_grid[record[3]:record[3] + record[5] + 1, record[2]:record[2] + record[4] + 1] = 1
            face_grids.append(np.reshape(face_grid, newshape=(25 * 25, 1)))

            # Reading target coordinates
            Y.append(np.reshape(np.array(record[6:]), newshape=(2)))

        return [np.array(right_eye_images), np.array(left_eye_images), np.array(face_images), np.array(face_grids)], np.array(Y)

    def on_epoch_end(self):

        # Shuffling record indices
        self.shuffled_idxes = np.arange(self.number_of_instances)
        if self.shuffle == True:
            np.random.shuffle(self.shuffled_idxes)