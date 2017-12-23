import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
import numpy as np


class Model(object):

    def __init__(self, config):
        self.config = config
        self.max_story_size = config['max_story_size']
        self.max_sentence_size = config['max_sentence_size']
        self.max_question_size = config['max_sentence_size']
        self.word_dim = config['word_dim']
        self.answer_dim = config['answer_dim']
        self.max_question_len = config['max_question_len']

        self.rnn_state_size_text = 3*self.word_dim
        self.rnn_state_size_question = 3*self.word_dim

        # place holders
        self.is_train = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, [None, self.max_story_size, self.max_sentence_size, self.word_dim])
        self.seq_length = tf.placeholder(tf.int32, [None, self.max_story_size])
        self.seq_length_qs = tf.placeholder(tf.int32, [None])
        self.questions = tf.placeholder(tf.float32, [None, self.max_question_len, self.word_dim])
        self.object_mask = tf.placeholder(tf.float32, [None, self.max_story_size, self.max_story_size])
        self.answers = tf.placeholder(tf.float32, [None, self.answer_dim])
        self.batch_size = tf.placeholder(tf.int32)
        self.build()

    def build(self):

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
            # }}}

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
                g_5 = fully_connected_layer(g_4, 256, name='g_5')
                return g_3  #change here

        def relation_network(objects, mask, q):
            all_g = []
            story_size = objects.shape[1]


            for i in range(story_size):
                o_i = objects[:, i]
                for j in range(story_size):
                    o_j = objects[:, j]
                    mask_ij = tf.reshape(mask[:, i, j], [self.batch_size, 1])
                    if i == 0 and j == 0:
                        g_i_j = g_theta(o_i, o_j, q, reuse=False)
                    else:
                        g_i_j = g_theta(o_i, o_j, q, reuse=True)
                    all_g.append(g_i_j * mask_ij)

            all_g = tf.stack(all_g, axis=0)
            all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
            output_relation_network = f_phi(all_g)
            return output_relation_network

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                fc_1 = fully_connected_layer(g, self.answer_dim+6, name='fc_1')
                fc_2 = fully_connected_layer(fc_1, self.answer_dim, name='fc_2')
                #fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=self.is_train, scope='fc_3/')
                fc_3 = fully_connected_layer(fc_2, self.answer_dim, activation_fn=tf.nn.relu, name='fc_3')
                return fc_3

        # X_reshaped[0] indicates all the first sentences in each story for all batch
        x_reshaped = tf.unstack(tf.transpose(self.x, perm=[1, 0, 2, 3]))  # [story_Size, batch_size,..,..]
        seq_length_reshaped = tf.unstack(tf.transpose(self.seq_length, perm=[1, 0]))
        # Get the final states : 2D censor , sentence embedding for each sentence per batch
        result = []
        is_first = True

        idx = 0
        for sentences in x_reshaped:
            with tf.variable_scope('lstm') as scope:
                if not is_first:
                    scope.reuse_variables()
                is_first = False
                lstm_cell = create_rnn_cell(self.rnn_state_size_text)
                outputs, states = tf.nn.dynamic_rnn(lstm_cell, sentences, dtype=tf.float32,
                                                    sequence_length=seq_length_reshaped[idx])
                idx = idx + 1
                result.append(states.h)

        # objects[i] indicates the representation of each sentence in the story of the i-th batch
        output = tf.transpose(tf.stack(result), perm=[1, 0, 2])

        lstm_cell_for_question = create_rnn_cell(self.rnn_state_size_question)
        question_embedding_outputs, question_embedding_states = tf.nn.dynamic_rnn(lstm_cell_for_question,
                                                                                  self.questions,
                                                                                  dtype=tf.float32,
                                                                                  sequence_length=self.seq_length_qs)
        self.output_rn = relation_network(output, self.object_mask, question_embedding_states.h)

        # verify whether the variables are reused
        for v in tf.global_variables():
            print(v.name)

        self.init = tf.global_variables_initializer()
        self.all_preds = tf.nn.softmax(self.output_rn)
        self.loss, self.accuracy = build_loss(self.output_rn, self.answers)

