# 《机器学习》张学工第二次实验
本实验的全部代码在 https://github.com/Baolar/ml_thu_course80250993 （目前仅自己可见）
本次实验参考 https://blog.csdn.net/u011119817/article/details/104602195
（一开始没参考死活调不出来，然后看了一遍然后手推一遍然后就自己写了）
exp5.X.py为实验代码（仅用numpy实现networks）
relu_by_pytorch.py是用pytorch写的一个relu作为激活函数的网络，作为对比参考
详细的实验数据都在实验报告中。

## 环境配置
常用环境配置，最新版本的numpy, pandas, scikit-learn, matplotlib等
python版本：python 3.8
```bash
pip install -r requirement.txt
```
本实验全部代码在Apple M1 Windows10 for ARM上运行通过

## 数据读取
和《实验一》一样，数据读取在data_loader.py内。只需要指定训练集和测试集的文件即可。
为了方便，训练集全部放在了/train下，测试集全部放在了/test下

## 数据预处理
调用sklearn的库，pca降至96维，同时标准化至0~1区间。

### 网络结构
```bash
Linear(96, 16)
Sigmoid()
Linear(16, 16)
Sigmoid()
Linear(16, 2)
```
在实验中，batch_size设置的为1。主要是电脑太慢了 && 实验室也没服务器 -.-
因为shufflesplit之后返回的是下标，如果要设置>1的batch_size需要新开一个数组去存training_set和test_set，需要耗费一定时间。。
（写readme的时候突然意识到这其实也废不了多少时间，o(n_splits*N)的复杂度而已。但已经来不及了。。。训练一次需要10小时。。）
（其实我也不知道为啥训练一次要这么久，用Adam理论上应该很快？）
（但是它就是用了这么久。。。从5-1中的学习曲线可以看到validation_score在3600epochs的时候才达到机制点，即使只训练3600epochs也得5个小时）

### 5.1 绘制学习曲线
lr=1e-2, epochs = 50000, optimizer = "Adam", cv =5, test_size=0.1
```bash
python exp5_1.py
```

### 5.2 输出test_set1上的正确率
lr=1e-2, epochs = 3600
```bash
python exp5_2.py
```
75.66%

### 5.3 根据你课上看过的ppt调参
更改之后的模型在exp5_3.py中
将Sigmoid换成了ReLU和LeakyReLU
但事实上这个模型是有问题的
1. exp会上溢出
2. 梯度更新有问题

至于有什么问题还在研究之中。
但5.1的模型已经可以了。比5000epochs pytorch写的正确率要高