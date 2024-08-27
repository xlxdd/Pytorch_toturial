import torch
import numpy as np
x = torch.empty(1)
print(x)
x=torch.empty(3)
print(x)
x=torch.empty(3,2)
print(x)

x=torch.rand(3,2)
print(x)
x=torch.zeros(3,2)
print(x)

x=torch.ones(2,2)
print(x)
print("打印类型")
print(x.dtype)
x=torch.ones(3,3,dtype=torch.double)
print(x.dtype)
x=torch.ones(3,3,dtype=torch.int)
print(x.dtype)

print(x.size())
print(x.size().numel())
x = torch.tensor([2.2,3.1])
print(x)
print("===========加法============")
x=torch.ones(3,3,dtype=torch.int)
print(x)
y = torch.ones(3,3,dtype=torch.int)
print(y)
z=x+y
print(z)
z=torch.add(x,y)
print(z)
print("===========计算print(y.add_(x))============")
print(y.add_(x)) #把x加到y中去
print("===========减法============")
z=x-y #矩阵相减
print(z)
z=torch.sub(x,y) #矩阵相减
print(z)
print("===========计算print(y.sub_(x))============")
print(y.sub_(x)) #把x减去从y中
print("===========乘法============") #这个乘法是元素相对的相乘，而不是线性代数的 A23*A32
z=x*y #矩阵相乘 
print(z)
z=torch.mul(x,y)
print(z)
print(y.mul_(x))
print("===========除法============")
z=x/y #矩阵相乘除
print(z)
z=torch.div(x,y)
print(z)
print("===========列表============")
x=torch.rand(5,4)  # 创建一个3*2的矩阵，并随机赋值
print(x[:,0]) #打印全部行，但只取第一列
print(x[0,:]) #打印全部列，但只取第一行
print(x[0,0]) #打印i=0 j=0的元素
print(x[1,1].item()) #如果只取一个元素值，则可以取他的真实值
print(x)
print("===========view可以resize tensor============")
x=torch.rand(5,4) 
y=x.view(20) #返回一个新的张量，这个是返回一个1行的20个元素的张量
print(y)
y=x.view(-1,10)
print(y) # 这个是返回2行，每行10个,他做了自动适配
print(y.size())#输出size
#print(x.view(-1,7)) # 这个自动适配不了，因为不能被7整除
print("===========numpy numpy只能在cpu上使用，不能在gpu上使用============")
a=torch.ones(5) #行向量，值是1，元素是5
b=a.numpy() #返回 numpy.ndarray类型的numpy下的张量，相当于转了类型，用于计算，该函数有参数 默认是false，表示使用cpu
print(b,type(b))
 #这里虽然a转了类型到b，但b和a是对象封装，引用地址一样 所以当我们给a+1时，b也会+1
a.add_(1)
print(a)
print(b)#虽然a b类型不一样，但值都改变了
print("===========从numpy.ndarray转成tensor张量的方式============")
a = np.ones(5) #行向量 5元素 值是1
b =torch.from_numpy(a) #numpy的ndarray转tensor 同样是装箱拆箱 修改a的值 b也会变
a+=1
print(b)
print(a)
print("===========gpu============")
if(torch.cuda.is_available()):
    #CUDA 是指 NVIDIA 的并行计算平台和编程模型，它利用图形处理单元 (GPU) 的处理能力来加速计算密集型任务
    device =torch.device("cuda") #获取cuda驱动
    x=torch.ones(5,device=device)#创建时指定了使用cpu的内存
    y=torch.ones(5)#创建时使用cpu的内存
    y=y.to(device)#将y转到gpu
    z=x+y #这个操作是在gpu的内存上进行了
    #z.numpy()#这个不能执行，因为z在gpu的内存上
    z =z.to("cpu") #转回到cpu
else:
    print("this is cpu")
