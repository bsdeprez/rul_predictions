import numpy as np

def generate_dataset(K, dataset_size):
    train_set, test_set = [], []
    for _ in range(dataset_size):
        generator = SinusoidGenerator(K)
        train_set.append(generator.batch())
        test_set.append(generator.equally_spaced_samples())
    return train_set, test_set


class SinusoidGenerator:

    def __init__(self, K=10, amplitude=None, phase=None):
        """
        :param K: batch size. Number of values sampled at every batch
        :param amplitude: Sine wave amplitude. If None, then the amplitude is uniformly sampled from [0.1, 5.0]
        :param phase: Sine wave phase. If None, then the phase is uniformly sampled from [0, π]
        """
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if amplitude else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self.__sample_x__()

    def __sample_x__(self):
        return np.random.uniform(-5, 5, self.K)

    def f(self, x):
        """Sine wave function"""
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x=None, force_new=False):
        """
        Returns a batch of size K.
        It also changes the shape of 'x' to add a batch dimension to it.
        :param x: Batch data, if given. 'y' is generated based on this data.
        :param force_new: Instead of using 'x' argument, the batch data is uniformly sampled
        """
        if x is None:
            if force_new:
                x = self.__sample_x__()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]

    def equally_spaced_samples(self, K=None):
        """:returns: K equally spaced samples."""
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))
