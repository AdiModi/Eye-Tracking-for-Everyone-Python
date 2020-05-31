import os
import warnings
import project_path as pp

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from keras.models import Model
    from keras.layers import Input, Conv2D, ZeroPadding2D, Dense, Flatten, MaxPool2D, Concatenate

# activation functions
activation = 'relu'
image_input_shape = (224, 224, 3)


# Eye Image Model
def get_eye_image_network():
    input_layer = Input(shape=image_input_shape, name='Eye_Image_Network_Input')

    layer = Conv2D(96, kernel_size=(11, 11), strides=4, activation=activation, name='conv-e1')(input_layer)
    layer = MaxPool2D(pool_size=(3, 3), strides=2)(layer)

    layer = ZeroPadding2D(padding=(2, 2))(layer)
    layer = Conv2D(256, kernel_size=(5, 5), strides=1, activation=activation, name='conv-e2')(layer)
    layer = MaxPool2D(pool_size=(3, 3), strides=2)(layer)

    layer = ZeroPadding2D(padding=(1, 1))(layer)
    layer = Conv2D(384, kernel_size=(3, 3), strides=1, activation=activation, name='conv-e3')(layer)

    layer = Conv2D(64, kernel_size=(1, 1), strides=1, activation=activation, name='conv-e4')(layer)

    output_layer = Flatten(name='Eye_Image_Network_Output')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.name = 'Eye_Image_Network'
    return model


# Face Image Model
def get_face_image_network():
    input_layer = Input(shape=image_input_shape, name='Face_Image_Network_Input')

    layer = Conv2D(96, kernel_size=(11, 11), strides=4, activation=activation, name='conv-f1')(input_layer)
    layer = MaxPool2D(pool_size=(3, 3), strides=2)(layer)

    layer = ZeroPadding2D(padding=(2, 2))(layer)
    layer = Conv2D(256, kernel_size=(5, 5), strides=1, activation=activation, name='conv-f2')(layer)
    layer = MaxPool2D(pool_size=(3, 3), strides=2)(layer)

    layer = ZeroPadding2D(padding=(1, 1))(layer)
    layer = Conv2D(384, kernel_size=(3, 3), strides=1, activation=activation, name='conv-f3')(layer)

    layer = Conv2D(64, kernel_size=(1, 1), strides=1, activation=activation, name='conv-f4')(layer)

    layer = Flatten(name='Face_Image_Network_Output')(layer)

    layer = Dense(128, activation=activation, use_bias=True, name='fc-e1')(layer)

    output_layer = Dense(64, activation=activation, use_bias=True, name='fc-e2')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.name = 'Face_Image_Network'
    return model


# Face Grid Model
def get_face_grid_network(grid_size=25):

    input_layer = Input(shape=(grid_size * grid_size, 1), name='Face_Grid_Network_Input')

    layer = Flatten()(input_layer)

    layer = Dense(256, activation=activation, use_bias=True, name='fc-fg1')(layer)

    output_layer = Dense(128, activation=activation, use_bias=True, name='fc-fg2')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.name = 'Face_Grid_Network'
    return model


# Final Model
def get_complete_model(save_summary=True):

    # get partial trained models
    eye_image_network = get_eye_image_network()
    # print(eye_image_network.summary())
    face_image_network = get_face_image_network()
    # print(face_image_network.summary())
    face_grid_network = get_face_grid_network(25)
    # print(face_grid_network.summary())

    # right eye model
    right_eye_image_network_input = Input(shape=image_input_shape, name='iTracker_Network_Right_Eye_Image_Input')
    right_eye_image_network_output = eye_image_network(right_eye_image_network_input)

    # left eye model
    left_eye_image_network_input = Input(shape=image_input_shape, name='iTracker_Network_Left_Eye_Image_Input')
    left_eye_image_network_output = eye_image_network(left_eye_image_network_input)

    # face model
    face_image_network_input = Input(shape=image_input_shape, name='iTracker_Network_Face_Eye_Image_Input')
    face_image_network_output = face_image_network(face_image_network_input)

    # face grid
    face_grid_network_input = Input(shape=(25 * 25, 1), name='iTracker_Network_Face_Grid_Input')
    face_grid_network_output = face_grid_network(face_grid_network_input)

    # dense layers for eyes
    eyes_conv_features = Concatenate(axis=-1)([left_eye_image_network_output, right_eye_image_network_output])
    fc_e1 = Dense(128, activation=activation, use_bias=True, name='fc-e1')(eyes_conv_features)

    # final dense layers
    merged_unique_features = Concatenate(axis=-1)([fc_e1, face_image_network_output, face_grid_network_output])
    fc1 = Dense(128, activation=activation, name='fc1')(merged_unique_features)
    fc2 = Dense(2, activation='linear', name='fc2')(fc1)

    # Complete (Final) Model
    model = Model(
        inputs=[right_eye_image_network_input, left_eye_image_network_input, face_image_network_input, face_grid_network_input],
        outputs=[fc2])
    # print(model.summary())

    model.name = 'iTracker_Network'

    if save_summary:
        model_summary_folder_path = os.path.join(pp.generated_folder_path, 'Model Summary')

        if not os.path.isdir(model_summary_folder_path):
            os.mkdir(model_summary_folder_path)

        with open(os.path.join(model_summary_folder_path, 'Eye Image Network.txt'), 'w') as file:
            eye_image_network.summary(print_fn=lambda line: file.write(line + '\n'))

        with open(os.path.join(model_summary_folder_path, 'Face Image Network.txt'), 'w') as file:
            face_image_network.summary(print_fn=lambda line: file.write(line + '\n'))

        with open(os.path.join(model_summary_folder_path, 'Face Grid Network.txt'), 'w') as file:
            face_grid_network.summary(print_fn=lambda line: file.write(line + '\n'))

        with open(os.path.join(model_summary_folder_path, 'iTracker Network.txt'), 'w') as file:
            model.summary(print_fn=lambda line: file.write(line + '\n'))

    return model

# get_complete_model(save_summary=True)