import visdom
import torch as t

#设定visdom的env
vis = visdom.Visdom(env='test1')

#visdom绘制曲线
x = t.arange(1,30,0.01)
y = t.sin(x)
vis.line(X=x,Y=y,win='sinx',opts={'title':'y=sin(x)'})

#visdom更新数值绘制曲线
for i in range(0,10):
    x=t.Tensor([i])
    y=x
    vis.line(X=x,Y=y,win='polynomial',update='append')

#visdom在原pane上增加曲线
x= t.arange(0,9,0.1)
y = (x**2)/9
vis.line(X=x,Y=y,win='polynomial',update='append',name="this is a new trace")


#visdom可视化图片
vis.image(t.randn(64,64).numpy(),win = 'random_img1')
vis.image(t.randn(3,64,64).numpy(),win = 'random_img2')
vis.images(t.randn(36,3,64,64).numpy(),nrow = 6,win = 'random_img3',opts={'title':'random_imgs'})
'''
这句不生效，说明line,img,text等不同类型不能共存在一个pane里
vis.line(X=x,Y=y,win='random_img3',update='append',name="this is a new trace")
'''