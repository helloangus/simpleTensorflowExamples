# 在终端中输入 tensorboard --logdir=logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加层函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random.normal([in_size, out_size]), name='W')
            tf.compat.v1.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.compat.v1.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.compat.v1.summary.histogram(layer_name+'/outputs', outputs)

        return outputs

# 生成训练数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # -1 to 1 ， 300 examples
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义输入placeholder，并在tensorboard中成组为input
with tf.name_scope('input'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_input')

# 生成含一层隐藏层的神经网络
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# 定义loss和优化器
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))    # 按行求和,再求各行的平均值
    tf.compat.v1.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化所有变量
init = tf.compat.v1.global_variables_initializer()

# 初始化对话
sess = tf.compat.v1.Session()
merged = tf.compat.v1.summary.merge_all()   # 合并所有summary
writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)    # 输出tensorboard信息
sess.run(init)

# 建立图形
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)  # 显示real data
plt.ion()   # 确保可以连续输入图形

# 进行训练
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        # 当有线的时候先抹除，否则跳过
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        # 创建新线
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        
        # 延迟绘图
        plt.pause(0.1)

# 显示现有图形不关闭
plt.show()
plt.pause(0)