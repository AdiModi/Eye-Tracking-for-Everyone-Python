import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import keras.backend as K

def euclidean_distance(y_estimated, y_actual):
    return K.sqrt(K.sum(K.pow(y_actual - y_estimated, 2), axis=1))

def max_euclidean_distance(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return K.max(ed)


def min_euclidean_distance(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return K.min(ed)


def mean_euclidean_distance(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return K.mean(ed)

def euclidean_distance_std(y_estimated, y_actual):
    ed = euclidean_distance(y_estimated, y_actual)
    return K.std(ed)
