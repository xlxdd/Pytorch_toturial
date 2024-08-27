import torch
print("===========requires_grad 例子1============")
#使用自动微分（autograd）
x=torch.ones(5,requires_grad=True) #默认requires_grad是false  1,计算梯度：requires_grad 是一个布尔参数，用于指定一个张量是否需要计算梯度 2,自动求导：使用 requires_grad=True 的张量进行的所有操作都会被记录，以便稍后使用 backward() 方法进行自动求导。
print(x)
# 对张量进行一些操作
y = x + 2
print(y)
# 再进行一些操作
z = y * y * 3
print("======分割线1=====")
print(z)
out = z.mean() #是一个张量操作，它计算张量 z 的所有元素的平均值。 看做f(x)=(x1+x2+x3+x4)/4 然后对每个x求偏导，在把值带回去
print("======分割线2=====")
print(out)
# 进行反向传播，计算梯度
out.backward()
print("======分割线3=====")
print(x.grad)  # 输出x的梯度
print("===========requires_grad 例子2============")
# 创建一个张量，并指定需要计算梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) 
# 定义一个标量函数 平方在求和 相当于函数 f()= x1²+ x2²,+ x3²
y = x.pow(2).sum() 
# 进行反向传播，计算梯度 x`的梯度，它对应于函数 f(x1,x2,x3)= x1²+ x2²+ x3², 三个偏导数就是 2x1,2x2,2x3,带入tensor的值 即'[2x1，2x2，2x3]
y.backward() 
# 输出 x 的梯度
print(x.grad)  # 输出 tensor([2., 4., 6.])
print("===========requires_grad 例子3============")
# 创建一个张量，并指定需要计算梯度
x = torch.tensor([[1.0, 2.0],[4.0, 5.0]], requires_grad=True) 
# 定义一个标量函数 平方在求和 相当于函数 f()= x1²+ x2²+ x3²+ x4²
y = x.pow(2).sum() 
# 进行反向传播，计算梯度 x`的梯度，这里是2*2矩阵，但计算的时候，就按元素个数算x，没有行列ij。
# 它对应于函数 f(x1,x2,x3,x4)= x1²+ x2²,+ x3²+ x4², 四个偏导数就是 2x1,2x2,2x3,2x4,带入tensor的值 即'[2x1，2x2，2x4,2x5]
y.backward() 
# 输出 x 的梯度
print(x.grad)  # 输出 tensor([[2, 4][8,10]])
