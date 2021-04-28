from tensorflow import keras
import tensorflow as tf

def loss_function(y_predicted, y_true):
    """
    The loss function used for every model. Returns the mean mse-loss.
    :param y_predicted: The predicted output.
    :param y_true: The expected output.
    :return: the mean mse-loss.
    """
    return keras.backend.mean(keras.losses.mean_squared_error(y_true, y_predicted))

def np_to_tensor(list_of_numpy_objects):
    """
    Converts a list of numpy objects to a list of tensorflow tensors.
    :param list_of_numpy_objects: The list of numpy objects.
    :return: A list of tensorflow tensors.
    """
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objects)
