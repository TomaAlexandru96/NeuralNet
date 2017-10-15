#!/usr/bin/env python

from neural_net import Net
import sys

def train(net, data):
    net.random_weights()
    net.start_session()
    net.print_net()
    net.train(data)
    net.save_to_file("part1_net")
    net.print_net()

n = Net(1, 10, 2, 1, 10)

if (sys.argv[1] == "train"):
    data = []
    for i in range(0, 10000):
        data.append([0, 0, 0])
    for i in range(0, 10000):
        data.append([0, 1, 0])
    for i in range(0, 10000):
        data.append([1, 0, 0])
    for i in range(0, 10000):
        data.append([1, 1, 1])
    train(n, data)
elif (sys.argv[1] == "solve"):
    data = [[float(sys.argv[2]), float(sys.argv[3])]]
    n.random_weights()
    n.weights_from_file("part1_net")
    res = n.propagate(data)
    print(res)
