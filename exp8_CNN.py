# 在终端中输入 tensorboard --logdir=logs    #

### import part ###
# 调整信息显示等级，越高信息越少
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1'  # 只使用第二块GPU

# 调用相关库
import tensorflow as tf
import shutil

# 调用数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('\n\n\nImport complete!\n')

### import part ###



### 训练参数定义    ###
input_size = 784
output_size = 10
init_learningRate = 0.0001 * 2
iteration = 1000
step_to_change_learningRate = 300
step_to_show_info = 100
step_to_record_info = 5

### 训练参数定义    ###



### def part    ###
# # 添加层函数
# def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
#     layer_name = 'layer%s' % n_layer
#     with tf.name_scope(layer_name):
#         with tf.name_scope('weights'):
#             Weights = tf.Variable(tf.random.normal([in_size, out_size]), name='W')
#             # tf.compat.v1.summary.histogram(layer_name+'/weights', Weights)
#         with tf.name_scope('biases'):    
#             biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#             # tf.compat.v1.summary.histogram(layer_name+'/biases', biases)
#         with tf.name_scope('Wx_plus_b'):
#             Wx_plus_b = tf.matmul(inputs, Weights) + biases
        
#         Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#         # tf.compat.v1.summary.histogram(layer_name+'/outputs', outputs)

#         return outputs

# 定义修改学习率函数
def change_learningRate(iteration, learningRate_op):  
    learningRate_op = learningRate_op / (tf.cast((iteration%step_to_change_learningRate < 1), tf.float32) + 1)  # 使用tf.cast((iteration%500 < 1), tf.float32)来判断已经达到一定的iteration（==不可用）
    return learningRate_op

# 定义计算准确率函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  # argmax按行比较，并输出最大值索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  # cast改变数据类型
    return accuracy

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # padding 'SAME'与原图片大小一致（包含零填充）  'VALID'比原图小
    return tf.compat.v1.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

### def part    ###



### placehouder part    ###
# 定义输入placeholder，并在tensorboard中成组为input
with tf.name_scope('input'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, input_size], name='x_input')
    ys = tf.compat.v1.placeholder(tf.float32, [None, output_size], name='y_input')
x_image = tf.reshape(xs, [-1,28,28,1])   # [num_of_samples, x, y, z]
# 学习率相关placholder
it = tf.compat.v1.placeholder(tf.float32, name='iteration')
lr = tf.compat.v1.placeholder(tf.float32, name='learningRate_op')

# 定义dropout中的保留比例
keep_prob = tf.compat.v1.placeholder(tf.float32)

### placehouder part    ###



### generate network    ###
# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])                    # [x, y, z, out_z]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)    # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             # output size 14*14*32

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])                   # [x, y, z, out_z]
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                             # output size 7*7*32

# func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
conv2_flat = tf.reshape(h_pool2, [-1, 7*7*64])              # [-1, 7, 7, 64] ->> [-1, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(conv2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

### generate network    ###



### define parameters   ###
# 定义学习率
with tf.name_scope('learning_rate'): 
    learningRate = tf.compat.v1.Variable(init_learningRate, name='learningRate')  # 用于更新学习率是暂存
    tf.compat.v1.summary.scalar('learning_rate', learningRate)
# 定义更新学习率的操作
learningRate_op = change_learningRate(it, lr)

# 定义cross_entropy和优化器、学习率
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.math.log(prediction), reduction_indices = [1]), name='cross_entropy')    # 按行求和,再求各行的平均值
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(learningRate_op).minimize(cross_entropy)

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
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 按需调节显存占用
# config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 强制显存占用为40%
sess = tf.compat.v1.Session(config=config)
sess.run(init)
# 合并并写入所有summary
merged = tf.compat.v1.summary.merge_all()   
path = 'logs'
shutil.rmtree(path) # 递归删除该目录下所有文件夹和文件
train_writer = tf.compat.v1.summary.FileWriter(path+'/train', sess.graph)    # 输出tensorboard信息
# test_writer = tf.compat.v1.summary.FileWriter(path+'/test', sess.graph)

print('Init complete!\n')

### initializer ###



### training step   ###
# 进行训练
print('\n\n\nTraining start!\n')

for i in range(iteration):
    # 进行每步训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate), keep_prob: 0.5})
    print('W_conv1', sess.run(W_conv1, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate), keep_prob: 0.5}))
    # 更新学习率
    new_value = sess.run(learningRate_op, feed_dict={it: i, lr: sess.run(learningRate)})
    update = tf.compat.v1.assign(learningRate, new_value)
    sess.run(update)
    # 打印相关信息
    if (i+1) % step_to_show_info == 0:
        print('iteration:     ', i+1)
        print('learning_rate: ', sess.run(learningRate))
        print('cross_entropy: ', sess.run(cross_entropy, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))
        print('accuracy:      ', sess.run(accuracy_op, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}), '\n')
    # 记录summary
    if (i+1) % step_to_record_info == 0:
        train_result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate), keep_prob: 1})
        # test_result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate), keep_prob: 1})
        train_writer.add_summary(train_result, i)
        # test_writer.add_summary(test_result, i)
        
print('Trainning finish!\n\n\n')

### training step   ###