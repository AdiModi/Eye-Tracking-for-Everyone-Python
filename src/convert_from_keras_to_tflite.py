import os
import project_path as pp
from custom_metrics import max_euclidean_distance, mean_euclidean_distance, min_euclidean_distance
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import tensorflow as tf

instance = '000'
instance_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + instance)
keras_models_folder_path = os.path.join(instance_folder_path, 'Keras')
tflite_models_folder_path = os.path.join(instance_folder_path, 'TFLite')

for folder_item in os.listdir(keras_models_folder_path):
    if folder_item.endswith('hdf5'):
        model_name = folder_item.replace('.hdf5', '')


        converter = tf.lite.TFLiteConverter.from_keras_model_file(os.path.join(keras_models_folder_path, folder_item),
                                                                  custom_objects={
                                                                      'min_deviation': min_euclidean_distance,
                                                                      'mean_deviation': mean_euclidean_distance,
                                                                      'max_deviation': max_euclidean_distance
                                                                  })
        tflite_model = converter.convert()
        open(os.path.join(tflite_models_folder_path, model_name + '.tflite'), 'wb').write(tflite_model)

        print('Converted "{}" Successfully!'.format(model_name))

print('Conversion Completed!')