#!/usr/bin/env python3

import random
import functools

data_file = "data.in"
n = 1000000
inputs = 5
outputs = 3

def rand():
    return random.uniform(0, 1)

with open(data_file, "w") as f:
    for i in range(0, n):
        ins = [rand() for i in range(0, inputs)]
        outs = [
                float(functools.reduce(lambda x,y: x + y, ins)) / float(inputs),
                float(functools.reduce(lambda x,y: x * y, ins)) / float(inputs),
                float(functools.reduce(lambda x,y: x * y, ins)) / float(inputs)
                ]
        string = ""
        for i in range(0, inputs):
            string += str(ins[i]) + "\t"
        for i in range(0, outputs):
            string += str(outs[i]) + "\t"
        string += "\n"
        f.write(string)
