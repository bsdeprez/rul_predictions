from RemainingUsefulLifeProject.Baseline.Models.FNNs.Baseline import FFNModel
from Unit_Testing.Helpers.plotter import plot_history, plot
from Unit_Testing.Helpers.sinusoidgenerator import generate_dataset
import matplotlib.pyplot as plt
import numpy as np

train_sets, test_sets = generate_dataset(1000, dataset_size=7)

# TEST THE MODEL ON A SINGLE SINE WAVE
train_x, train_y = train_sets[0]
test_x, test_y = test_sets[0]

model_1 = FFNModel(1)
history = model_1.train(train_x, train_y, epochs=150)
predicted = model_1.predict(test_x)

plot_history(history)
plot((test_x, test_y), label="True values")
plot((test_x, predicted), label="Predicted values")
plt.legend(loc="upper right")
plt.show()

# TEST THE MODEL ON MULTIPLE SINE WAVES
# Expected: The model will converge to the mean of the different sine waves
for index in range(1, len(train_sets)):
    x, y = train_sets[index]
    train_x, train_y = np.concatenate((train_x, x)), np.concatenate((train_y, y))
    x, y = test_sets[index]
    test_x, test_y = np.concatenate((test_x, x)), np.concatenate((test_y, y))
model_2 = FFNModel(1)
history = model_2.train(train_x, train_y, epochs=150)
predicted = model_2.predict(test_x)

plot_history(history)
for i, test_set in enumerate(test_sets):
    plot(test_set, label="Sine {}".format(i))
plt.scatter(test_x, predicted, label="Predicted")
plt.legend(loc="upper right")
plt.show()

# COMPARE THIS PREDICTION TO THE AVERAGE
# Calculate average prediction
avg_pred = []
for test_set in test_sets:
    x, y = test_set
    avg_pred.append(y)
avg_plot, = plt.plot(x, np.mean(avg_pred, axis=0), '--')
predicted = model_2.predict(x)
model_plot, = plt.plot(x, predicted)
plt.legend([avg_plot, model_plot], ['Average', 'Standard FFN'])
plt.show()
