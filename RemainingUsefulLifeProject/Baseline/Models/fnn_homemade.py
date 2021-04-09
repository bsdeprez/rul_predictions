import tensorflow as tf


class FNNModel:

    def __init__(self):
        n_inputs = 28*28
        n_hidden1 = 300
        n_hidden2 = 100
        n_output = 10

        n_iterations = 50
        n_batches = 33
        learn_rate = 0.00003

        X = tf.placeholder(dtype=tf.float32, shape = [None, n_inputs], name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')

        with tf.name_scope("myfnn"):
            h1 = self.__create_layer__(X, n_hidden1, layer_name="hidden layer 1", activation_fcn=tf.nn.relu)
            h2 = self.__create_layer__(h1, n_hidden2, layer_name="hidden layer 2", activation_fcn=tf.nn.relu)
            logits = self.__create_layer__(h2, n_output, layer_name='output')

        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(entropy)

    @staticmethod
    def __create_layer__(input_layer, n_neurons, layer_name="", activation_fcn=None):
        with tf.name_scope(layer_name):
            n_inputs = int(input_layer.get_shape()[1])
            initial_value = tf.truncated_normal((n_inputs, n_neurons))
            w = tf.Variable(initial_value, name="weight")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            op = tf.matmul(input_layer, w) + b
            if activation_fcn:
                op = activation_fcn(op)
            return op
