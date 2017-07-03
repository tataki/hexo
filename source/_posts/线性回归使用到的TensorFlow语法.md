---
title: 线性回归使用到的TensorFlow语法
date: 2017-07-03 18:19:48
tags: TensorFlow
categories: TensorFlow
---

##基础语法部分

```python
import tensorflow as tf

#-----------准备阶段-----------
a = tf.Variable([[2,3]])
b = tf.Variable([[4],[2]])
c = tf.matmul(a,b)

print('c------->',c)

#创建用0填充的矩阵
d = tf.zeros([2,4])
#平方
e = tf.square([2])
#平均值
f = tf.reduce_mean([1,3])
#均匀分布的随机数
q = tf.random_uniform([1,10])
#------------执行阶段-----------
with tf.Session() as sess:
	#初始所有的变量
	init = tf.global_variables_initializer()
	sess.run(init)

```

##代码实现部分

```python
# -*- coding: UTF-8 -*-
# 一元的线性回归模型的训练
# 1.通过训练数据推测出线性回归函数（y = w * x + b）中的w 和 b的值
# 2.通过验证数据，验证得到的函数是否符合预期

#引入tensorflow
import tensorflow as tf
#引入绘图标

#引入数据模块
import testData as td

# 1.获得训练数据
# testData
# get_train_data 获得训练数据，参数data_length(获得数据的个数)返回：二维数组[0]代表x [1]代表y
# get_validata_data 获得验证数据 参数：data_length(数据个数) 返回二维数组 二维数组[0]代表x [1]代表y

trainData = td.get_train_data(200)
trainx = [v[0] for v in trainData]
trainy = [v[1] for v in trainData]

#2.构造预测的线性回归函数 y = w * x +b
w = tf.Variable(tf.random_uniform([1]))
b = tf.Variable(tf.zeros([1]))
y = w * trainx + b

#3.判断假设函数的好坏
# 代价函数

cost = tf.reduce_mean(tf.square(y-trainy))

#4.调整假设函数
#梯度下降算法找最优解
optimizer = tf.train.GradientDescentOptimizer(0.08)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    #--------初始化所有变量值--------
    init = tf.initialize_all_variables() #replaced initialize_all_variables with global_variables_initializer
    sess.run(init)
    #初始化w.b的值
    print("cost=",sess.run(cost),"w=",sess.run(w),"b=",sess.run(b))
    #循环运行
    for k in range(1000):
        sess.run(train)
        #输出训练好的w和b
        print("cost=",sess.run(cost),"w=",sess.run(w),"b=",sess.run(b))
    print("执行完成")

    #构造图形结构
```
