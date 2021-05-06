import matplotlib.pyplot as plt

def plot(data, *args, **kwargs):
    """Plot helper"""
    x, y = data
    return plt.plot(x, y, *args, **kwargs)

def plot_history(history):
    plt.title("Training history")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(loc='upper right')
    plt.show()
