from Linear_practice import Linear
import torch.nn as nn
import torch as t


class Perceptron(nn.Module):
    def __init__(self,in_features,hidden_features,out_features):
        super(Perceptron, self).__init__()
        self.layer1 = Linear(in_features,hidden_features)
        self.layer2 = Linear(hidden_features,out_features)

    def forward(self,x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        x = self.layer2(x)
        return x

perceptron = Perceptron(3,4,1)
for name,parameter in perceptron.named_parameters():
    print(name,':',parameter.size())