import matplotlib.pyplot as plt
import tensorflow as tf
from sinusoidgenerator import SinusoidGenerator, plot
from mamlmodel import train_maml
from transfermodel import *
from sinemodel import SineModel
import numpy as np


def generate_dataset(K, train_size=20000, test_size=10):
    """
    Generate train and test dataset.
    A dataset is composed of SinusoidGenerators that are able to provide a batch of 'K' elements at a time.
    """

    def __generate_dataset__(size):
        return [SinusoidGenerator(K=K) for _ in range(size)]

    return __generate_dataset__(train_size), __generate_dataset__(test_size)

def plot_model_comparison_to_average(model, ds, model_name='neural network', K=10):
    """
    Compare model to average.

    Computes mean of training sine waves actual 'y' and compare the model's prediction to a new sine wave,
    the intuition is that these two plots should be similar.
    """
    sinus_generator = SinusoidGenerator(K)
    # Calculate average prediction
    avg_pred = []
    for i, sinusoid_generator in enumerate(ds):
        x, y = sinusoid_generator.equally_spaced_samples()
        avg_pred.append(y)
    x, _ = sinus_generator.equally_spaced_samples()
    avg_plot, = plt.plot(x, np.mean(avg_pred, axis=0), '--')

    # Calculate model prediction
    model_pred = model.forward(tf.convert_to_tensor(x))
    model_plot, = plt.plot(x, model_pred.numpy())

    # Plot
    plt.legend([avg_plot, model_plot], ['Average', model_name])
    plt.show()


train_ds, test_ds = generate_dataset(K=10)
neural_net = SineModel()
train_maml(neural_net, 1, train_ds)
for index in np.random.randint(0, len(test_ds), size=3):
    eval_sinewave_for_test(neural_net, test_ds[index])