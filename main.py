#!/usr/bin/env python

from neural_net import Net
import sys

n = Net()

def train(data):
    n.random_weights()
    n.start_session()
    n.print_net()
    n.train(data)
    n.save_to_file("part1_net")
    n.print_net()

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
    train(data)
elif (sys.argv[1] == "solve"):
    data = [[float(sys.argv[2]), float(sys.argv[3])]]
    n.random_weights()
    n.weights_from_file("part1_net")
    res = n.propagate(data)
    print(res[0])
