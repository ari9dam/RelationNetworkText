import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn

n_steps = 2
word_size = 3
rnn_state_size = 5
answer_dimension = 2
is_train = True


def create_rnn_cell(size):  # Create a cell that should be reused
    basic_cell = rnn.LSTMCell(num_units=size)
    return basic_cell


def fully_connected_layer(input, output_shape, activation_fn=tf.nn.relu, name="fc"):
    fcl = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    return fcl


def g_theta(o_i, o_j, q, reuse, scope='g_theta'):
    with tf.variable_scope(scope, reuse=reuse) as scope:

        g_1 = fully_connected_layer(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
        g_2 = fully_connected_layer(g_1, 256, name='g_2')
        g_3 = fully_connected_layer(g_2, 256, name='g_3')
        g_4 = fully_connected_layer(g_3, 256, name='g_4')
        return g_4


def relation_network(objects, story_size, q):
    all_g = []
    for i in range(story_size):
        o_i = objects[i]
        for j in range(objects):
            o_j = objects[j]

            if i == 0 and j == 0:
                g_i_j = g_theta(o_i, o_j, q, reuse=False)
            else:
                g_i_j = g_theta(o_i, o_j, q, reuse=True)
            all_g.append(g_i_j)

    all_g = tf.stack(all_g, axis=0)
    all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
    output = f_phi(all_g)

def f_phi(g, scope='f_phi'):
    with tf.variable_scope(scope) as scope:
        fc_1 = fully_connected_layer(g, 256, name='fc_1')
        fc_2 = fully_connected_layer(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
        fc_3 = fully_connected_layer(fc_2, answer_dimension, activation_fn=None, name='fc_3')
        return fc_3

# 1D array indicating the number of sentences in each story
# story_length = tf.placeholder(tf.int32,[None])
# 1D array indicating the length of each sentence
# seq_length = tf.placeholder(tf.int32,[None])

# data that each rnn receives
# The 1st argument says the number of <x,y> pairs in each batch
#   which is set as None denoting that it can accept any number of samples
# The 2nd argument stands for story size
# The 3rd argument denotes the length of sequence for each instance
# The 4th argument denotes the word embedding size of each token in the sequence
# [batch size, story size, seq_length, word_size]
X = tf.placeholder(tf.float32, [None, None, None, word_size])
# X_reshaped[0] indicates all the first sentences in each story for all batch
X_reshaped = tf.unstack(tf.transpose(X,perm=[1,0,2,3]))  # [story_Size, batch_size,..,..]

# Get the final states : 2D censor , sentence embedding for each sentence per batch
result = []
for sentences in X_reshaped:
    with tf.variable_scope('lstm') as scope:
        scope.reuse_variables()
        lstm = create_rnn_cell(5)
        output_vals, states = tf.contrib.rnn.dynamic_rnn(lstm, sentences, dtype=tf.float32)
        result.append(states.h)

# objects[i] indicates the representation of each sentence in the story of the i-th batch
output = tf.transpose(tf.stack(result), perm=[1,0,2])

# create relation network
