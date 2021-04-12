import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.autograd import Variable as V
t.manual_seed(1)


def f(x):
    y = x ** 2 * t.exp(x)
    return y

def gradf(x):
    dx = 2*x*t.exp(x)+x**2*t.exp(x)
    return dx

x = V(t.randn(3, 4), requires_grad=True)
y = f(x)

y.backward(t.ones(y.size()))    #由于y不是标量，因此求后向传播时必须传入与其形状一样的参数,此处值为1的原因详见书《pytorch入门与实践》P83页，我也没太明白
print(x.grad)

print(gradf(x))