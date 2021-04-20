import torch as t
import torch.nn as nn

net = nn.Module()
#net.param = t.randn(2,2)
net.param1 = nn.Parameter(t.randn(3,3))
print(net.param1)
net.load_state_dict(t.load('save_module/net.pth'))



for name,parameters in net.named_parameters():
    print(name,parameters)