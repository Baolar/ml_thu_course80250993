# 清华大学《机器学习》——张学工作业
课程号：80250993-200 
这是 course exercise 1的readme
All the code is available on: https://github.com/Baolar/ml_thu_course80250993/tree/main/course_exercise_1
目前设置为private
将在作业成绩公布以后设置public

## 实验结果
识别结果存放在**test1_results.csv**与**test2_results.csv**中
其存的是在这四种方法中，本人实验的正确率最高的结果(PCA后LR）。总正确率为79.57%
n_iter=1000, lr=1e-3, n_splits=10, optimizer="Adam"
```bash
python main.py
```
对生死判别影响最大的主成分排序结果在**feature_weight.csv**中
每一题的实验结果都在报告中

## 环境配置
将训练集和测试集放入在了.\train与.\test下。

所需环境：python3.8 和几个最新版本的包
```bash
conda create -n env_name python=3.8
conda activate env_name
pip install -r requirement.txt
```
## 读取数据
实现data_loader.py
调用Xdata_loader.ex1_data()即可
其对应参数为:
array of trainingset X path
array of trainingset label path
array of testset X path
array of testset label path
例: 用train1与train2训练，test1与test2测试
```python
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv', 'train2_icu_data.csv'],\
        ['train1_icu_label.csv','train2_icu_label.csv'],\
        ['test1_icu_data.csv', 'test2_icu_data.csv'],\
        ['test1_icu_label.csv', 'test2_icu_label.csv'])
```

## 模型
三种模型均为手写，最多调用numpy库（不会np都不让用吧），knn和pca直接调sklearn的库。模型封装在.\models.py下。

## 实验一 FID
PCA参数"mle"
$\hat{m}$阈值的参数为$-\frac{1}{2} * (\hat{m_1} + \hat{m_2})$

```bash
python exp1.py 
```

## 实验二 Perception
### 2.1 
lr=1e-2, n_split=5, n_iter=1000
由于添加了cross-validation，故此测试过程较慢，两种分类器都跑一遍大约需要10分钟。
```bash
python exp2_1.py
```
### 2.2 
lr=1e-2, n_iter=1000, n_splits=5
和2.1类似
```bash
python exp2_2.py
```

### 2.3
n_split=5, n_iter=3000, r=5e-3
```bash
python exp2_3.py
```

### 2.4
n_split=5, n_iter=3000, r=5e-3
```bash
python exp2_4.py
```

## 实验三 logistic regression
### 3.1 and 3.2
n_iter=1000, 1e-2, n_splits=5, optimizer=“Adam”
```bash
python exp3_1and3_2.py
```

### 3.3
n_iter=500, lr=1e-2, n_splits=5, optimizer="Adam"
```bash
python exp3_3.py
```

### 3.4
```bash
python exp3_4.py
```
结果一部分显示在控制台，全部的主成分排序结果写入至{\$workspace}\feature_weight.csv

## 实验四 knn
### 4.1~4.3 
代码中的for循环是为了找到最合适的k的值。
我（每次写个报告用“我”显得很不正式用“我们”又觉得怪怪的）是先画了一类weight的图再改一个weight的值画一个图。
控制台输出预测结果

```bash
python exp4_1to4_3.py
```
