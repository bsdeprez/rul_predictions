from time import sleep

import keras
import random
import tensorflow as tf
from tqdm import tqdm


def to_tensor_slices(x, y):
    return tf.data.Dataset.from_tensor_slices((x, y))


def train_maml(model, epochs, dataset_list, batch_size=100, lr_inner=0.01, lr_outer=0.03):
    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.MeanSquaredError()
    # STEP 2
    for epoch in range(1, epochs + 1):
        total_loss = 0
        losses = []
        # STEP 3 and 4
        for i, dataset in enumerate(random.sample(dataset_list, len(dataset_list))):
            train_data = dataset.shuffle(buffer_size=100).batch(batch_size)
            with tqdm(
                    desc="Epochs[{:0>3d}/{:0>3d}] Dataset [{:0>2d}/{:0>2d}]".format(epoch, epochs, i, len(dataset_list))
                    , total=len(train_data)) as progressbar:
                for x, y in train_data:
                    model.forward(x)  # Run forward pass to initialize weights
                    with tf.GradientTape() as test_tape:
                        # STEP 5
                        with tf.GradientTape() as train_tape:
                            train_loss, _ = model.compute_loss(x, y, loss_fn)
                        # STEP 6
                        gradients = train_tape.gradient(train_loss, model.trainable_variables)
                        k = 0
                        model_copy = model.copy_model(x)
                        for j in range(len(model_copy.layers)):
                            model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                                                      tf.multiply(lr_inner, gradients[k]))
                            model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
                                                                    tf.multiply(lr_inner, gradients[k+1]))
                            k += 2
                        # STEP 7
                        test_loss, logits = model_copy.compute_loss(x, y, loss_fn)
                    # STEP 8
                    gradients = test_tape.gradient(test_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    progressbar.update()
