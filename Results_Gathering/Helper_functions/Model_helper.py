import keras
import keras.backend as K

def loss_function(pred_y, y):
    return K.mean(keras.losses.mean_squared_error(y, pred_y))


