import sys
import math

import tensorflow as tf
from random import shuffle

class Net:
    weights = []
    biases = []
    hidden_layers_nr = 2
    neurons_per_layer = 30
    nr_of_inputs = 7
    nr_of_outputs = 3
    batch_size = 10

    def __init__(self):
        pass

    def random_weights(self):
        self.init_random_weights()
        self.init_propagate_step()

    def weights_from_file(self, file):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, file)

    def init_random_weights(self):
        add_bias_weight = lambda l1, l2: (self.weights.append(tf.Variable(tf.random_normal([l1, l2]))), \
                self.biases.append(tf.Variable(tf.random_normal([1, l2]))))

        add_bias_weight(self.nr_of_inputs, self.neurons_per_layer)
        for i in range(self.hidden_layers_nr-1):
            add_bias_weight(self.neurons_per_layer, self.neurons_per_layer)
        add_bias_weight(self.neurons_per_layer, self.nr_of_outputs)

    def init_propagate_step(self):
        self.input = tf.placeholder(tf.float32, [None, self.nr_of_inputs])
        self.output = tf.placeholder(tf.float32, [None, self.nr_of_outputs])

        self.input = tf.nn.l2_normalize(self.input, 1)

        self.actual_out = self.input

        for i in range(self.hidden_layers_nr+1):
            z = tf.add(tf.matmul(self.actual_out, self.weights[i]), self.biases[i])
            self.actual_out = tf.nn.sigmoid(z)

        diff = tf.subtract(self.actual_out, self.output)
        cost = tf.multiply(diff, diff)
        self.step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    def start_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, training_set):
        shuffle(training_set)
        for i in range(0, len(training_set), self.batch_size):
            mat_in = []
            mat_out = []
            for batch in range(i, min(len(training_set), i+self.batch_size)):
                tokens = [float(x) for x in training_set[batch].split()]
                mat_in.append(tokens[:self.nr_of_inputs])
                mat_out.append(tokens[self.nr_of_inputs:])
            self.sess.run(self.step, feed_dict = {self.input: mat_in, self.output: mat_out})

    def propagate(self, input):
        return self.sess.run(self.actual_out, feed_dict = {self.input: input})

    def save_to_file(self, name):
        saver = tf.train.Saver()
        saver.save(self.sess, name)

    def print_net(self):
        for w in self.weights:
            print(self.sess.run(w))
            print()
        for b in self.biases:
            print(self.sess.run(b))
            print()
