import tensorflow.keras.backend as K
import keras
import random
import tensorflow as tf


def np_to_tensor(list_of_numpy_objects):
    return (tf.convert_to_tensor(obj) for obj in list_of_numpy_objects)


def loss_function(pred_y, y):
    return K.mean(keras.losses.mean_squared_error(y, pred_y))


def compute_loss(model, x, y, loss_fn=loss_function):
    logits = model(x)
    mse = loss_fn(y, logits)
    return mse, logits


def train_maml(model, copy_function, epochs, dataset, lr_inner=0.01):
    optimizer = keras.optimizers.Adam()
    for epoch in range(1, epochs + 1):
        print("Epoch {:0>3d} of {:0>3d}".format(epoch, epochs))
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = t
            with tf.GradientTape() as test_tape:
                with tf.GradientTape() as train_tape:
                    train_loss, _ = compute_loss(model.model, x, y)
                gradients = train_tape.gradient(train_loss, model.model.trainable_variables)
                k = 0
                model_copy = copy_function(model)
                for j in range(len(model_copy.model.layers)):
                    model_copy.model.layers[j].kernel = tf.subtract(model.model.layers[j].kernel,
                                                                    tf.multiply(lr_inner, gradients[k]))
                    model_copy.model.layers[j].bias = tf.subtract(model.model.layers[j].bias,
                                                                  tf.multiply(lr_inner, gradients[k + 1]))
                    k += 2
                test_loss, logits = compute_loss(model_copy.model, x, y)
            gradients = test_tape.gradient(test_loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))
    return model
