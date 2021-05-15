from RemainingUsefulLifeProject.Baseline.Models.FNNs.Baseline import FFNModel, copy_model
from Unit_Testing.Helpers.evaluators import eval_sinewave_for_test
from Unit_Testing.Helpers.sinusoidgenerator import generate_dataset, SinusoidGenerator
import numpy as np

train_sets, test_sets = generate_dataset(1000, dataset_size=7)

model = FFNModel(1)
# STEP 1: Train the neural network on a couple of datasets.
train_x, train_y = train_sets[0]
test_x, test_y = test_sets[0]
for index in range(1, len(test_sets)):
    x, y = train_sets[index]
    train_x, train_y = np.concatenate((train_x, x)), np.concatenate((train_y, y))
    x, y = test_sets[index]
    test_x, test_y = np.concatenate((test_x, x)), np.concatenate((test_y, y))
history = model.train(train_x, train_y, epochs=150)

# STEP 2: Apply transfer learning to the model
generators = [SinusoidGenerator(K=10) for _ in range(7)]
for index in np.random.randint(0, len(generators), size=3):
    eval_sinewave_for_test(model, copy_fn=copy_model, sinusoid_generator=generators[index])
