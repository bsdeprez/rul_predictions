from Unit_Testing.Helpers.sinusoidgenerator import SinusoidGenerator
import keras
import numpy as np
import matplotlib.pyplot as plt

def eval_sine_test(model, x, y, x_test, y_test, num_steps=(0, 1, 10)):
    """
    Evaluate how the model fits to the curve training for 'fits' steps.
    :param model: Model evaluated.
    :param x: Data used for training.
    :param y: Targets used for training.
    :param x_test: Data used for evaluation.
    :param y_test: Targets used for evaluation.
    :param num_steps: Number of steps to log.
    """
    fit_res = []
    if 0 in num_steps:
        loss, logits = model.compute_loss(x_test, y_test)
        fit_res.append((0, logits, loss))

    for step in range(1, np.max(num_steps)+1):
        model.train(x, y, epochs=1)
        loss, logits = model.compute_loss(x_test, y_test)
        if step in num_steps:
            fit_res.append((step, logits, loss))

    return fit_res

def eval_sinewave_for_test(model, copy_fn, sinusoid_generator=None, num_steps=(0, 1, 10), line_style='--', plot=True, title=""):
    """
    Evaluates how the sinewave adapts at dataset.
    The idea is to use the pretrained model as a weight initializer and try to fit the model on this new dataset.

    :param model: Already trained model
    :param copy_fn: The function that copies the model.
    :param sinusoid_generator: A sinusoidGenerator instance.
    :param num_steps: Number of training steps to be logged.
    :param lr: Learning rate used for training steps to be logged.
    :param line_style: The style of the line used to in the plot.
    :param plot: If plot is True, than it plots how the curves are fitted along 'num_steps'.
    :param title: The title of the plot.
    :returns: The fit results. A list containing the loss, logits and step,  for every step at 'num_steps'.
    """

    if sinusoid_generator is None:
        sinusoid_generator = SinusoidGenerator(K=10)

    # Generate equally spaced samples for plotting
    x_test, y_test = sinusoid_generator.equally_spaced_samples(100)

    # batch used for training
    x, y = sinusoid_generator.batch()

    # Copy model so we can use the same model multiple times
    copied_model = copy_fn(model)

    # run training and log fit results
    fit_res = eval_sine_test(copied_model, x, y, x_test, y_test, num_steps)

    # plot
    train, = plt.plot(x, y, '^')
    ground_truth, = plt.plot(x_test, y_test)
    plots = [train, ground_truth]
    legend = ['Training Points', 'True Function']
    for n, res, loss in fit_res:
        cur, = plt.plot(x_test.flatten(), res[:, 0], line_style)
        plots.append(cur)
        legend.append(f'After {n} Steps')
    plt.legend(plots, legend)
    plt.ylim(-5, 5)
    plt.xlim(-6, 6)
    plt.title(title)
    if plot: plt.show()
    else: plt.close()
    return fit_res
