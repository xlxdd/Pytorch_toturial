import torch
print("============randn==============")
a = torch.randn(3) #这里是randn 不是rand  torch.randn：生成服从标准正态分布（均值为0，标准差为1）的随机数。  torch.rand：生成服从均匀分布（在区间 [0, 1) 之间）的随机数。
print(a)
b=a+2
print(b) # 输出tensor是a+2计算后的结果
x = torch.randn(3,requires_grad=True) #这里是randn 不是rand
print(x)
y=x+2
print(y)  # 输出tensor是x+2计算后的结果，同时记录了函数，grad_fn=<AddBackward0> 表示是加法函数 grad=gradient  fn=Function
print("============求梯度1==============")
x = torch.randn(3,requires_grad=True)
print(x)
y=x+2
z=y*y*2
print(z) # 这里会增加属性，grad_fn=<MulBackward0> ，这里的mul表示是乘法
z=z.mean() # 指定标量函数
#这里必须指定标量函数，如果删除z=z.mean() 这句话会提示 grad can be implicitly created only for scalar outputs 
print(z) # 属性grad_fn=<MeanBackward0>，Mean表示平均值函数
z.backward() #逆向传播 如果requires_grad=False，则执行z.backward()回抛异常，因为没有记录grad_fn
print(x.grad)
print("============清零grad==============")
weights =torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output =(weights*3).sum()#设置标量值，这里是连写了，分开就是a=weight*3 model_output=a.sum()
    model_output.backward()
    print(weights.grad)
    model_output.zero_()#可以注释这一行，看看不清零的效果
print("============加权==============")
x = torch.randn(3,requires_grad=True)
print(x)
y=x+2
z=y*y*2
v = torch.tensor([1.0,2.0,3.0],dtype=torch.float32)
z.backward(v)
print(x.grad)