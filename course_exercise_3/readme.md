# 《机器学习》张学工第二次实验
本实验的全部代码在 https://github.com/Baolar/ml_thu_course80250993 （目前仅自己可见）
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

### 6.2 对比不同参数的影响
对比linear、ploy、rbf不同kernel对结果的影响。交叉验证使用ShuffleSplite库函数，test_size设置为0.1，cv=5
同时对比C的不同值对实验的影响
```bash
python exp6_2.py
```

### 6.3 输出预测结果
选取实验6.2中最好的参数， C=1，kernel="linear"
```bash
python exp6_3.py
```
79.3%