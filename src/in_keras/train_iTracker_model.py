import os
import project_path as pp
from keras.models import load_model
from in_keras.dataloader import DataBatcher
from in_keras import iTracker_model
from in_keras.custom_metrics import min_euclidean_distance, mean_euclidean_distance, max_euclidean_distance, euclidean_distance_std
import json
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from keras.callbacks import ModelCheckpoint
    from keras import optimizers


def main():
    # Hyper Parameters
    hyper_paramaters = {
        'pretrained_model_file_path': None,
        # 'pretrained_model_file_path': os.path.join(pp.trained_models_folder_path, 'Instance_002', 'keras', 'Model-030-5.438.hdf5'),
        'NUM_EPOCHS': 30,
        'BATCH_SIZE': 150,
        'LEARNING_RATE': 0.01
    }

    # Creating instance folder to save models
    instance = 1
    while os.path.isdir(os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))):
        instance += 1
    instance_folder_path = os.path.join(pp.trained_models_folder_path, 'Instance_' + str(instance).zfill(3))
    os.mkdir(instance_folder_path)
    hyper_parameters_file_path = os.path.join(instance_folder_path, 'Hyper Parameters.json')
    keras_models_folder_path = os.path.join(instance_folder_path, 'keras')
    tflite_models_folder_path = os.path.join(instance_folder_path, 'tflite')

    # Saving Hyper Parameters
    with open(os.path.join(hyper_parameters_file_path), 'w') as  file:
        json.dump(hyper_paramaters, file)

    # Creating directories for model
    os.mkdir(keras_models_folder_path)
    os.mkdir(tflite_models_folder_path)

    # Loading Data Generators
    training_generator = DataBatcher(batch_size=hyper_paramaters['BATCH_SIZE'], type=DataBatcher.TRAIN)
    validation_generator = DataBatcher(batch_size=hyper_paramaters['BATCH_SIZE'], type=DataBatcher.VALIDATION)

    # Design optimizer
    # optimizer = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.99, amsgrad=1e-08)
    optimizer = optimizers.Adagrad(lr=hyper_paramaters['LEARNING_RATE'])

    if hyper_paramaters['pretrained_model_file_path'] == None or \
            not (os.path.isfile(hyper_paramaters['pretrained_model_file_path']) and hyper_paramaters[
        'pretrained_model_file_path'].endswith('.hdf5')):
        model = iTracker_model.get_complete_model(save_summary=True)
        model.compile(optimizer=optimizer,
                      loss='mse',
                      metrics=[min_euclidean_distance, mean_euclidean_distance, max_euclidean_distance,
                               euclidean_distance_std])
    else:
        model = load_model(hyper_paramaters['pretrained_model_file_path'], custom_objects={
            'min_euclidean_distance': min_euclidean_distance,
            'mean_euclidean_distance': mean_euclidean_distance,
            'max_euclidean_distance': max_euclidean_distance,
            'euclidean_distance_std': euclidean_distance_std
        })
        print('Loaded model using:', hyper_paramaters['pretrained_model_file_path'])

    # Saving Checkpoints
    trained_model_file_path = os.path.join(keras_models_folder_path, 'Model-{epoch:03d}-{val_loss:.3f}.hdf5')
    checkpoint = ModelCheckpoint(trained_model_file_path,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 mode='auto',
                                 period=1)

    callbacks_list = [checkpoint]

    # Train model on dataset
    print('Commencing Training')
    model.fit_generator(generator=training_generator,
                        epochs=hyper_paramaters['NUM_EPOCHS'],
                        validation_data=validation_generator,
                        callbacks=callbacks_list,
                        # use_multiprocessing=True,
                        # max_queue_size=2,
                        verbose=1,
                        workers=2)
    print('Training Completed!')

if __name__ == '__main__':
    main()