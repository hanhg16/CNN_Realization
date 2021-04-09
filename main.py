import torch as t



a = t.Tensor(5,3)
b = t.rand(5,3)
print(a+b)

a= a.cuda()
b=b.cuda()

print(a+b)

