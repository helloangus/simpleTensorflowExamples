import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf

state = tf.Variable(0, name = 'counter')
print(state.name)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.compat.v1.assign(state, new_value)    # 将new_value加载到state上  

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))