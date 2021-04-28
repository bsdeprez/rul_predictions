import keras
import time
import random
import matplotlib.pyplot as plt
from sinemodel import np_to_tensor
import tensorflow as tf
from transfermodel import copy_model


def train_maml(model, epochs, dataset, lr_inner=0.01, batch_size=1, log_steps=1000):
    """
    Train using the MAML setup.
    :param model: A model.
    :param epochs: Number of epochs used for training.
    :param dataset: A dataset used for training.
    :param lr_inner: Inner learning rate. Default value is 0.01.
    :param batch_size: Batch size. Default value is 1.
    :param log_steps: At every 'log_steps', a log message is printed.
    :return: A strong, fully-developed and trained maml.
    """
    optimizer = keras.optimizers.Adam()
    # STEP 2: instead of checking for convergence, we train for a number of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        # STEP 3 and 4
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            x, y = np_to_tensor(t.batch())
            model.forward(x) # run forward pass to initialize weights
            with tf.GradientTape() as test_tape:
                # STEP 5
                with tf.GradientTape() as train_tape:
                    train_loss, _ = model.compute_loss(x, y)
                # STEP 6
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                k = 0
                model_copy = copy_model(model, x)
                for j in range(len(model_copy.layers)):
                    model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                                              tf.multiply(lr_inner, gradients[k]))
                    model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients[k+1]))
                    k += 2
                # STEP 7
                test_loss, logits = model_copy.compute_loss(x, y)
            # STEP 8
            gradients = test_tape.gradient(test_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Logs
            total_loss += test_loss
            loss = total_loss / (i + 1.0)
            losses.append(loss)

            if i % log_steps == 0 and i > 0:
                print("Step {}: loss = {}, Time to run {} steps = {}".format(
                    i, loss, log_steps, time.time() - start
                ))
                start = time.time()
        plt.plot(losses)
        plt.show()
