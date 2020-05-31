import warnings
import project_path as pp
import os
import numpy as np
from custom_metrics import euclidean_distance
from dataloader_keras import DataBatcher

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf


def euclidean_distance(y_estimated, y_actual):
    return np.sqrt(np.sum(np.power(y_actual - y_estimated, 2), axis=1))


def max_deviation(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return np.max(ed)


def min_deviation(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return np.min(ed)


def mean_deviation(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return np.mean(ed)


def std(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return np.std(ed)


test_generator = DataBatcher(type=DataBatcher.TEST, batch_size=500)
number_of_batches = len(test_generator)

instance = '014'
instance_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + instance)
tflite_models_folder_path = os.path.join(instance_folder_path, 'TFLite')

folder_items = os.listdir(tflite_models_folder_path)
folder_items.reverse()

for item in folder_items:
    if not item.endswith('.tflite'):
        continue

    tflite_model_file_path = os.path.join(tflite_models_folder_path, item)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    batch_number = 1
    number_of_instances = 0

    mean_dev = 0
    min_dev = float('inf')
    max_dev = -1 * float('inf')

    print('-' * 80)
    print('Processing Model:', item)
    for i in range(number_of_batches):
        X, Y = test_generator[i]
        batch_size = Y.shape[0]

        right_eyes = X[0]
        left_eyes = X[1]
        faces = X[2]
        face_grids = X[3]

        actual_Y = []
        tflite_predicted_Y = []

        print('\tProcessing Batch {} of {}...'.format(batch_number, number_of_batches))

        for j in range(batch_size):
            interpreter.set_tensor(input_details[0]['index'],
                                   np.asarray(np.expand_dims(right_eyes[j], axis=0), dtype=np.float32))
            interpreter.set_tensor(input_details[1]['index'],
                                   np.asarray(np.expand_dims(left_eyes[j], axis=0), dtype=np.float32))
            interpreter.set_tensor(input_details[2]['index'],
                                   np.asarray(np.expand_dims(faces[j], axis=0), dtype=np.float32))
            interpreter.set_tensor(input_details[3]['index'],
                                   np.asarray(np.expand_dims(face_grids[j], axis=0), dtype=np.float32))

            interpreter.invoke()

            predicted_Y = interpreter.get_tensor(output_details[0]['index'])
            tflite_predicted_Y.append(predicted_Y[0])
            actual_Y.append(Y[j])

        tflite_predicted_Y = np.asarray(tflite_predicted_Y)
        actual_Y = np.asarray(actual_Y)

        if min_dev > min_deviation(tflite_predicted_Y, actual_Y):
            min_dev = min_deviation(tflite_predicted_Y, actual_Y)

        if max_dev < max_deviation(tflite_predicted_Y, actual_Y):
            max_dev = max_deviation(tflite_predicted_Y, actual_Y)

        mean_dev = ((mean_dev * number_of_instances) + (mean_deviation(tflite_predicted_Y, actual_Y) * batch_size)) / (number_of_instances + batch_size)
        number_of_instances += batch_size

        print('\tMin Deviation:', min_dev)
        print('\tMean Deviation:', mean_dev)
        print('\tMax Deviation:', max_dev)
        print()

        batch_number += 1

    print('-' * 80)
    print('Min Deviation:', min_dev)
    print('Mean Deviation:', mean_dev)
    print('Max Deviation:', max_dev)
    print('-' * 80)
