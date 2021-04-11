import tensorflow as tf


class FNNModel:

    def __init__(self, n_inputs, hidden_layers=None):

        if hidden_layers is None:
            hidden_layers = [24, 5, 3]

        print("GPU available? {}".format(tf.test.is_gpu_available()))

    @staticmethod
    def __create_layer__(input_layer, n_neurons, layer_name="", activation_fcn=None):
        with tf.name_scope(layer_name):
            n_inputs = int(input_layer.get_shape()[1])
            initial_value = tf.random.truncated_normal((n_inputs, n_neurons))
            w = tf.Variable(initial_value, name="weight")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            op = tf.matmul(input_layer, w) + b
            if activation_fcn:
                op = activation_fcn(op)
            return op

    @staticmethod

