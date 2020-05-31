# 在终端中输入 tensorboard --logdir=logs    #

### import part ###
# 调整信息显示等级，越高信息越少
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# 调用相关库
import tensorflow as tf
import shutil

# 调用数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('\n\n\nImport complete!\n')

### import part ###



### def part    ###
# 添加层函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
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

# 定义修改学习率函数
def change_learningRate(iteration, learningRate_op):  
    learningRate_op = learningRate_op / (tf.cast((iteration%500 < 1), tf.float32) + 1)  # 使用tf.cast((iteration%500 < 1), tf.float32)来判断已经达到一定的iteration（==不可用）
    return learningRate_op

# 定义计算准确率函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  # argmax按行比较，并输出最大值索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  # cast改变数据类型
    return accuracy

### def part    ###



### placehouder part    ###
# 定义输入placeholder，并在tensorboard中成组为input
with tf.name_scope('input'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 784], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 10], name='y_input')
# 学习率相关placholder
it = tf.compat.v1.placeholder(tf.float32, name='iteration')
lr = tf.compat.v1.placeholder(tf.float32, name='learningRate_op')

### placehouder part    ###



### generate network    ###
# 生成含一层隐藏层的神经网络
l1 = add_layer(xs, 784, 400, n_layer=1, activation_function=tf.nn.sigmoid)
prediction = add_layer(l1, 400, 10, n_layer=2, activation_function=tf.nn.softmax)   # softmax常用于classification

### generate network    ###



### define parameters   ###
# 定义学习率
with tf.name_scope('learning_rate'): 
    learningRate = tf.compat.v1.Variable(0.8, name='learningRate')  # 用于更新学习率是暂存
    tf.compat.v1.summary.scalar('learning_rate', learningRate)
# 定义更新学习率的操作
learningRate_op = change_learningRate(it, lr)

# 定义cross_entropy和优化器、学习率
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.math.log(prediction), reduction_indices = [1]), name='cross_entropy')    # 按行求和,再求各行的平均值
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.GradientDescentOptimizer(learningRate_op).minimize(cross_entropy)

# 定义accuracy
with tf.name_scope('accuracy'):
    accuracy_op = compute_accuracy(xs, ys)
    tf.compat.v1.summary.scalar('accuracy', accuracy_op)

print('Define complete!\n')

### define parameters   ###



### initializer ###
# 初始化所有变量
init = tf.compat.v1.global_variables_initializer()
# 初始化对话
sess = tf.compat.v1.Session()
sess.run(init)
# 合并并写入所有summary
merged = tf.compat.v1.summary.merge_all()   
path = 'logs'
shutil.rmtree(path)    # 递归删除该目录下所有文件夹和文件
writer = tf.compat.v1.summary.FileWriter(path, sess.graph)    # 输出tensorboard信息

print('Init complete!\n')

### initializer ###



### training step   ###
# 进行训练
print('\n\n\nTraining start!\n')

for i in range(1500):
    # 进行每步训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate)})
    # 更新学习率
    new_value = sess.run(learningRate_op, feed_dict={it: i, lr: sess.run(learningRate)})
    update = tf.compat.v1.assign(learningRate, new_value)
    sess.run(update)
    # 打印相关信息
    if (i+1) % 500 == 0:
        print('iteration:     ', i+1)
        print('learning_rate: ', sess.run(learningRate))
        print('cross_entropy: ', sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys}))
        print('accuracy:      ', sess.run(accuracy_op, feed_dict={xs: batch_xs, ys: batch_ys}), '\n')
    # 记录summary
    if (i+1) % 50 == 0:
        result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate)})
        writer.add_summary(result, i)

print('Trainning finish!\n\n\n')
        
### training step   ###