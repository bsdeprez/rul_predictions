from RemainingUsefulLifeProject.Baseline.Models.Baseline import FFNModel, copy_model
from RemainingUsefulLifeProject.MAML.Models.maml import train_maml
from Unit_Testing.Helpers.evaluators import eval_sinewave_for_test
from Unit_Testing.Helpers.sinusoidgenerator import generate_dataset, SinusoidGenerator
import numpy as np

model = FFNModel(1)

train_sets, test_sets = generate_dataset(1000, dataset_size=7)
train_x, train_y = train_sets[0]
test_x, test_y = test_sets[0]
for index in range(1, len(test_sets)):
    x, y = train_sets[index]
    train_x, train_y = np.concatenate((train_x, x)), np.concatenate((train_y, y))
    x, y = test_sets[index]
    test_x, test_y = np.concatenate((test_x, x)), np.concatenate((test_y, y))
copied_model = copy_model(model)
copied_model.train(train_x, train_y, epochs=500)

maml = train_maml(model, copy_model, epochs=500, dataset=train_sets)

generators = [SinusoidGenerator(K=10) for _ in range(7)]
for index in np.random.randint(0, len(generators), size=3):
    eval_sinewave_for_test(copied_model, copy_fn=copy_model, sinusoid_generator=generators[index], title="Transfer learning")
    eval_sinewave_for_test(maml, copy_fn=copy_model, sinusoid_generator=generators[index], title="MAML")
