import os
import project_path as pp
from in_keras.dataloader import DataBatcher
from in_keras.custom_metrics import min_euclidean_distance, mean_euclidean_distance, max_euclidean_distance
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from keras.models import load_model

# Hyper Parameters
keras_model_file_path = os.path.join(pp.trained_models_folder_path, 'Instance_002', 'keras', 'Model-030-5.438.hdf5')
BATCH_SIZE = 250

# Generators
test_generator = DataBatcher(batch_size=BATCH_SIZE, type=DataBatcher.TEST)

# Loading model
model = load_model(keras_model_file_path,
                   custom_objects={'min_deviation': min_euclidean_distance,
                                   'mean_deviation': mean_euclidean_distance,
                                   'max_deviation': max_euclidean_distance})

mean_dev = 0
min_dev = float('inf')
max_dev = -1 * float('inf')
number_of_instances = 0
batch_number = 1
number_of_batches = len(test_generator)

for i in range(len(test_generator)):
    print('Processing Batch {} of {}...'.format(batch_number, number_of_batches))

    batch = test_generator[i]
    batch_size = test_generator[i][1].shape[0]
    predictions = model.test_on_batch(test_generator[i][0], test_generator[i][1])
    if predictions[1] < min_dev:
        min_dev = predictions[1]

    if predictions[3] > max_dev:
        max_dev = predictions[3]

    mean_dev = ((mean_dev * number_of_instances) + (predictions[2] * batch_size)) / (number_of_instances + batch_size)
    number_of_instances += batch_size
    batch_number += 1

    if batch_number % 10 == 0:
        print('Min Deviation:', min_dev)
        print('Mean Deviation:', mean_dev)
        print('Max Deviation:', max_dev)


print('Min Deviation:', min_dev)
print('Mean Deviation:', mean_dev)
print('Max Deviation:', max_dev)