import matplotlib.pyplot as plt
import torch as t
t.manual_seed(1000)

def get_fake_data(batch_size=8):
    x = t.rand(batch_size,1)*20
    y = x*2 + (1+t.randn(batch_size,1))*3
    return x,y

#随机初始化参数
w = t.rand(1,1)
b = t.zeros(1,1)
lr = 0.001

for i in range(20000):
    x,y = get_fake_data()

    #计算loss
    y_pred = x*w + b
    loss = 0.5*(y_pred - y)**2
    loss = loss.sum()

    # 手动求梯度
    dw = ((y_pred - y)*x).sum()
    db = (y_pred - y).sum()

    # 更新参数
    w.sub_(lr*dw)
    b.sub_(lr*db)

    # 绘图
    if i%1000 == 0:
        x = t.arange(0,20).view(-1,1)
        y = x*w + b
        plt.plot(x,y)   以x,y坐标点绘制曲线

        x2,y2 = get_fake_data(batch_size=20)
        plt.scatter(x2,y2)  添加x,y坐标散点


        plt.xlim(0,20)  #设置x轴范围
        plt.ylim(-5,45)
        plt.draw()  #可以在更新数据后在原图上直接刷新，在大部分情况下与plt.show()没区别，但在此处可以让程序继续执行而不需要等到手动关闭图像；以及，由于会自动更新，因此若不加最后的  plt.close()  ，所有绘图都会显示在一张图片上
        plt.pause(3) #暂停3s
        plt.close()

        print(w,b)