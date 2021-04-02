import matplotlib.pyplot as plt

def plot_difference(y_true, y_predicted):
    d = y_predicted - y_true
    plt.hist(d)
    plt.show()

def plot_predicted_v_true(y_true, y_predicted):
    plt.scatter(y_true, y_predicted, c='crimson')
    p1 = max(max(y_predicted), max(y_true))
    p2 = min(min(y_predicted), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.gca().invert_xaxis()
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.show()
