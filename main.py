#!/usr/bin/env python3

from neural_net import Net
import sys
import math

save_file = "./net"
data_file = "./data.in"

def train(net, data):
    net.random_weights()
    net.start_session()
    net.print_net()
    net.train(data)
    net.save_to_file(save_file)
    net.print_net()

# Net(hidden_layers, nr_of_neurons_per_layer, inputs, outputs, batch)
inputs = 5
outputs = 3
n = Net(1, 10, inputs, outputs, 10)

if (sys.argv[1] == "train"):
    data = []
    with open(data_file) as f:
        for line in f.readlines():
            line_data = line.strip().split("\t")
            data.append(list(map(lambda x: float(x), line_data)))
    train(n, data)
elif (sys.argv[1] == "solve"):
    data = [[float(sys.argv[i]) for i in range(2, 2+inputs)]]
    n.weights_from_file(save_file)
    res = n.propagate(data)
    print(res)
