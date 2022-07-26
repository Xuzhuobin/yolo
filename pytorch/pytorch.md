# 第一章 train

## 1.1 模型参数初始化

![image-20220726105040004](C:\Users\DELL\AppData\Roaming\Typora\typora-user-images\image-20220726105040004.png)



## 1.2 数据、模型保存和读取

1.state_dict：

![image-20220726144722429](C:\Users\DELL\Desktop\YOLO笔记\pytorch\pytorch.assets\image-20220726144722429.png)

说明：

```python
modelName.state_dict() # 得到模型中可学习的所有参数
optimizerName.state_dict() # 得到优化器的状态和使用的超参数

```



### 1.2.1 数据的保存、加载

```python
保存
torch.save(model.state_dict(),"PATH")

加载
model = modleClass  # 实例化modlel类
model.load_state_dict(torch.load("PATH"))
```



### 1.2.2 模型的保存、加载

```python
保存
torch.save(model,"PATH")

加载
model = torch.load("PATH")
```



## 1.3 GPU计算

### 1.3.1 单GPU训练

1.查看GPU状态

```
nvidia-smi
```

2.计算设备

![image-20220726154030989](C:\Users\DELL\Desktop\YOLO笔记\pytorch\pytorch.assets\image-20220726154030989.png)



3.TENSOR的GPU计算

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else
'cpu') 

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device) 
x
```



4.模型的GPU计算

```python
model.to(device)
```



### 1.3.2 多GPU训练

1.多GPU计算

```python
model = torch.nn.DataParallel(model, device_ids=[0,1])
```



2.多GPU模型的保存与加载

```python
torch.save(modelName.module.state_dict(),"PATH")
```



## 1.4 2d和1d

说明：一维向量(如全连接层)使用Conv1d、BatchNorm1d；高阶张量(如卷积层)使用Conv2d、BatchNorm2d。