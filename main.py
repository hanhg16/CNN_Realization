import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch.autograd import Variable as V

x = V(t.ones(1))
b = V(t.rand(1), requires_grad=True)
w = V(t.rand(1), requires_grad=True)
y = w * x
z = b + y


z.backward(retain_graph=True
           )
print(t.autograd.grad(y,w))
print(w.grad)


