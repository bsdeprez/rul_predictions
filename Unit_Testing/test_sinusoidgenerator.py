from Unit_Testing.Helpers.plotter import plot
from Unit_Testing.Helpers.sinusoidgenerator import SinusoidGenerator, generate_dataset
import matplotlib.pyplot as plt

generator = SinusoidGenerator(K=100)
plt.title('Test sinusoid generator')
plot(generator.equally_spaced_samples(), label="Test equally spaced samples")
x, y = generator.batch()
plt.scatter(x, y, label="Test batch", color="orange")
plt.legend(loc="upper right")
plt.show()

train_sets, test_sets = generate_dataset(100, 3)
colors = ['blue', 'orange', 'green']
for i, test_set in enumerate(test_sets):
    plt.title("Test generate dataset")
    plot(test_set, label="Sine {} test".format(i+1), color=colors[i])
    x, y = train_sets[i]
    plt.scatter(x, y, label="Sine {} train".format(i+1), color=colors[i])
plt.legend(loc="upper right")
plt.show()
