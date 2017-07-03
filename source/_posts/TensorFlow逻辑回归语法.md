---
title: TensorFlow逻辑回归语法
date: 2017-07-03 21:26:15
tags: TensorFlow
categories: TensorFlow
---

!["softmax"](https://ggg.9170.gs/hexo/img/softmax.png)

### 语法部分

```python
import tensorflow as tf
import numpy as np

#占位符，适用于不知道具体参数
x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.add(x, x)

argmax_paramter = tf.Variable([[1, 32, 44, 56],[89, 12, 90, 33],[35, 69,1,10]])

#最大列索引
argmax_0 = tf.argmax(argmax_paramter, 0)
#最大行索引
argmax_1 = tf.argmax(argmax_paramter, 1)

#平均数
reduce_0 = tf.reduce_mean(argmax_paramter, reduction_indices=0)
reduce_1 = tf.reduce_mean(argmax_paramter, raduction_indices=1)

#相等
equal_0 = tf.equal(1,2)
equal_1 = tf.equal(2, 2)

#类型转换
cast_0 = tf.cast(equal_0, tf.int32)
casr_1 = tf.cast(equal_1,tf.float32)

with tf.Session() as sess:
	init = tf.variables_all_initializer()
	sess.run(init)
	
	rand_array = np.random.rand(4, 4)
	print(sess.run(y, feed_dict={x: rand_array}))
```