#!/usr/bin/env python

from neural_net import Net
import sys
import math

def train(net, data):
    net.random_weights()
    net.start_session()
    net.print_net()
    net.train(data)
    net.save_to_file("part1_net")
    net.print_net()

# Net(hidden_layers, nr_of_neurons_per_layer, inputs, outputs, batch)
n = Net(1, 10, 2, 1, 10)

if (sys.argv[1] == "train"):
	data = []
	for i in range(0, 10000):
		data.append([1, 62, 63])
		data.append([2, 15, 17])
		data.append([9, 10, 19])
		data.append([11, 21, 32])
	train(n, data)
elif (sys.argv[1] == "solve"):
    data = [[float(sys.argv[2]), float(sys.argv[3])]]
    n.weights_from_file("part1_net")
    res = n.propagate(data)
    print(res)
