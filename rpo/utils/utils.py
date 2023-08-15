from collections import namedtuple

Type = namedtuple("Type", ["shape", "dtype"])

def max_grad(net):
    max_val = 0
    for weight in net.parameters():
        max_val = max(max_val, weight.grad.max())

    return max_val