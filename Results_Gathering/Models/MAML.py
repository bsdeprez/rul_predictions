import random
import keras
import tensorflow as tf


def train_maml(model, copy_function, epochs, dataset, lr_inner):
    optimizer = keras.optimizers.Adam()
    raw_model = model.get_model()
    for epoch in range(1, epochs + 1):
        print("Epoch {:0>3d} of {:0>3d}".format(epoch, epochs))
        for i, data in enumerate(random.sample(dataset, len(dataset))):
            x, y = data
            with tf.GradientTape() as test_tape:
                with tf.GradientTape() as train_tape:
                    train_loss, _ = model.compute_loss(x, y)
                gradients = train_tape.gradient(train_loss, raw_model.trainable_variables)
                k = 0
                model_copy = copy_function(model, x)
                raw_copy = model_copy.get_model()
                for j in range(len(raw_copy.layers)):
                    raw_copy.layers[j].kernel = tf.subtract(raw_copy.layers[j].kernel,
                                                            tf.multiply(lr_inner, gradients[k]))
                    raw_copy.layers[j].bias = tf.subtract(raw_copy.layers[j].bias,
                                                          tf.multiply(lr_inner, gradients[k+1]))
                    k += 2
                test_loss, logits = model_copy.compute_loss(x, y)
            gradients = test_tape.gradient(test_loss, raw_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, raw_model.trainable_variables))
    return model

