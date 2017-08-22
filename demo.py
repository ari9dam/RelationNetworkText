import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
import numpy as np

rnn_state_size = 5
word_size = 2
answer_dimension = 2
is_train = tf.placeholder(tf.bool)


def create_rnn_cell(size):  # Create a cell that should be reused
    basic_cell = rnn.LSTMCell(size)
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


def relation_network(objects, mask, q):
    all_g = []
    story_size = objects.shape[1]
    cur_batch_size = mask.shape[0]

    print(story_size)

    for i in range(story_size):
        o_i = objects[:,i]
        for j in range(story_size):
            o_j = objects[:,j]
            mask_ij = tf.reshape(mask[:,i,j],[2,1]) #np.array(np.array(mask[:,i,j])[np.newaxis].T)#tf.reshape(mask[:,i,j],[cur_batch_size,1])

            print(o_i.shape)
            if i == 0 and j == 0:
                g_i_j = g_theta(o_i, o_j, q, reuse=False)
            else:
                g_i_j = g_theta(o_i, o_j, q, reuse=True)
            all_g.append(g_i_j*mask_ij)

    all_g = tf.stack(all_g, axis=0)
    all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
    output_relation_network = f_phi(all_g)
    return  output_relation_network


def f_phi(g, scope='f_phi'):
    with tf.variable_scope(scope) as scope:
        fc_1 = fully_connected_layer(g, 256, name='fc_1')
        fc_2 = fully_connected_layer(fc_1, 256, name='fc_2')
        fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
        fc_3 = fully_connected_layer(fc_2, answer_dimension, activation_fn=tf.nn.relu, name='fc_3')
        return fc_3


# [batch size, story size, seq_length, word_size]
X = tf.placeholder(tf.float32, [None, 4, 3, word_size])
seq_length = tf.placeholder(tf.int32, [None,4])
Questions = tf.placeholder(tf.float32,[None,rnn_state_size])
objectmask = tf.placeholder(tf.float32, [None, 4, 4])
# X_reshaped[0] indicates all the first sentences in each story for all batch
X_reshaped = tf.unstack(tf.transpose(X,perm=[1,0,2,3]))  # [story_Size, batch_size,..,..]
seq_length_reshaped = tf.unstack(tf.transpose(seq_length,perm=[1,0]))
# Get the final states : 2D censor , sentence embedding for each sentence per batch
result = []
isFirst = True

idx = 0
for sentences in X_reshaped:
    with tf.variable_scope('lstm') as scope:
        if not isFirst:
            scope.reuse_variables()
        isFirst = False
        lstm_cell = create_rnn_cell(5)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, sentences, dtype=tf.float32,
                                            sequence_length=seq_length_reshaped[idx])
        idx = idx+1
        result.append(states.h)


# objects[i] indicates the representation of each sentence in the story of the i-th batch
output = tf.transpose(tf.stack(result), perm=[1,0,2])

lstm_cell_for_question = create_rnn_cell(5)
question_embedding_outputs, question_embedding_states  = tf.nn.dynamic_rnn(lstm_cell_for_question, sentences,
                                                                           dtype=tf.float32)
output_rn = relation_network(output,objectmask,question_embedding_states.h)

# verify whether the variables are reused
for v in tf.global_variables():
    print(v.name)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    mask = []
    mask.append(np.ones([4,4],dtype=float))
    s2 = np.zeros([4,4],dtype=float)
    s2[0:2,0:2] = np.ones([2,2])
    mask.append(s2)
    mask = np.array(mask)

    seq_len = np.array([[3,2,3,0],[1,1,0,0]])
    output_val, output_rn_val = sess.run([output,output_rn] ,
                                         feed_dict={X: np.random.rand(2,4,3,2), objectmask:mask,
                                                    Questions:np.random.rand(2,5), seq_length:seq_len, is_train:False})
    print(output_val)
    print(output_rn_val)

    output_val, output_rn_val = sess.run([output, output_rn],
                                         feed_dict={X: np.random.rand(2, 4, 3, 2), objectmask: mask,
                                                    Questions: np.random.rand(2, 5), seq_length: seq_len,
                                                    is_train: True})

    print(output_val)
    print(output_rn_val)
