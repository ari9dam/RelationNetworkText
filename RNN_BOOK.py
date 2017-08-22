import tensorflow as tf
import  numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

X_batch = np.random.rand(4,2,3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    outputs_val,states_val = sess.run([outputs,states],feed_dict={X: X_batch})

print(outputs_val)
print("states")
print(states_val.h)