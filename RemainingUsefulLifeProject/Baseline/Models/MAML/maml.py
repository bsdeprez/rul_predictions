from time import sleep

import keras
import random
import tensorflow as tf
from tqdm import tqdm

def to_tensor_slices(x, y):
    return tf.data.Dataset.from_tensor_slices((x, y))

def train_maml(model, epochs, dataset_list, batch_size=100):
    optimizer = keras.optimizers.Adam()
    # STEP 2
    for epoch in range(1, epochs+1):
        with tqdm(desc="[{:0>3d}/{:0>3d}]".format(epoch, epochs), total=len(dataset_list)) as progressbar:
            total_loss = 0
            losses = []
            # STEP 3 and 4
            for dataset in random.sample(dataset_list, len(dataset_list)):
                for x_batch, y_batch in dataset:
                    print(type(x_batch))

