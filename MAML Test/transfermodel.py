from sinemodel import SineModel, np_to_tensor
from sinusoidgenerator import SinusoidGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras


def copy_model(model, x):
    """
    Copy model weights to a new model.

    :param model: model to be copied
    :param x: An input example. This is used to run a forward pass in order to add the weights of the graph as variables
    :return: A copy of the model.
    """
    copied_model = SineModel()
    copied_model.forward(tf.convert_to_tensor(x))
    copied_model.set_weights(model.get_weights())
    return copied_model


def eval_sine_test(model, optimizer, x, y, x_test, y_test, num_steps=(0, 1, 10)):
    """
    Evaluate how the model fits to the curve training for 'fits' steps.
    :param model: Model evaluated.
    :param optimizer: Optimizer to be for training.
    :param x: Data used for training.
    :param y: Targets used for training.
    :param x_test: Data used for evaluation.
    :param y_test: Targets used for evaluation.
    :param num_steps: Number of steps to log.
    """
    fit_res = []
    tensor_x_test, tensor_y_test = np_to_tensor((x_test, y_test))
    # If 0 in fits, we log the loss before any training
    if 0 in num_steps:
        loss, logits = model.compute_loss(tensor_x_test, tensor_y_test)
        fit_res.append((0, logits, loss))
    for step in range(1, np.max(num_steps) + 1):
        model.train_batch(x, y, optimizer)
        loss, logits = model.compute_loss(tensor_x_test, tensor_y_test)
        if step in num_steps:
            fit_res.append(
                (step, logits, loss)
            )
    return fit_res


def eval_sinewave_for_test(model, sinusoid_generator=None, num_steps=(0, 1, 10), lr=0.01, plot=True):
    """
    Evaluate how the sinewave adapts at dataset.
    :param model: Already trained model.
    :param sinusoid_generator: A sinusoidGenerator instance.
    :param num_steps: Number of training steps to be logged.
    :param lr: Learning rate used for training on the test data.
    :param plot: If plot is True than it plots how the curves are fitted along 'num_steps'.
    :return: The fit results. A list containing the loss, logits and step for every step at 'num_steps'.
    """
    if sinusoid_generator is None:
        sinusoid_generator = SinusoidGenerator(K=10)
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)
    x, y = sinusoid_generator.batch()
    copied_model = copy_model(model, x)
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    fit_res = eval_sine_test(copied_model, optimizer, x, y, x_test, y_test, num_steps)

    train, = plt.plot(x, y, '^')
    ground_truth, = plt.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(x_test, res[:, 0], '--')
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    if plot:
        plt.show()
    return fit_res
