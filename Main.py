import data_reader
import relation_network as rn
import math
import os
from util import log
import time
import tensorflow as tf
from pprint import pprint



def log_step_message(step, accuracy, accuracy_test, loss, step_time, is_train=True):
    if step_time == 0:
        step_time = 0.001
    log_fn = (is_train and log.info or log.infov)
    log_fn((" [{split_mode:5s} step {step:4d}] " +
            "Loss: {loss:.5f} " +
            "Accuracy train: {accuracy:.2f} "
            "Accuracy test: {accuracy_test:.2f} " +
            "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
            ).format(split_mode=(is_train and 'train' or 'val'),
                     step=int(step),
                     loss=loss,
                     accuracy=accuracy*100,
                     accuracy_test=accuracy_test*100,
                     sec_per_batch=step_time,
                     instance_per_sec=15 / step_time
                     )
           )

# training and testing starts here
training_file = 'task21_blocksWorld_training.txt'
test_file = 'task21_blocksWorld_test.txt'

# temp = test_file
# test_file = training_file
# training_file  = temp

# read the training data
stories,tstories, questions, tquestions, answers, tanswers, seq_lens,tseq_lens, questions_len,tquestions_len,\
object_mask,tobject_mask = data_reader.process_data(training_file,test_file)

batch_size = 15

# compute number of batches
total_examples = answers.shape[0]
number_of_batches = math.ceil(total_examples /batch_size)

# create the model
config = {
    'max_story_size' : stories.shape[1],
    'max_sentence_size' : stories.shape[2],
    'word_dim' : stories.shape[3],
    'answer_dim' :  stories.shape[3],
    'max_question_len' : questions.shape[1]
}
rnmodel = rn.Model(config)

# create the optimizer
global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

learning_rate = tf.train.exponential_decay(
    0.001,
    global_step=global_step,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True,
    name='decaying_learning_rate'
)

optimizer = tf.contrib.layers.optimize_loss(
    loss=rnmodel.loss,
    global_step=global_step,
    learning_rate=learning_rate,
    optimizer=tf.train.AdamOptimizer,
    clip_gradients=20.0,
    name='optimizer_loss'
)

# train the model
print("Training Starts!")
session = tf.Session()

max_epoch = 400

output_save_step = 1000
start_index = 0
end_index = batch_size

init_main = tf.global_variables_initializer()
# initialize the variables in the model
session.run(rnmodel.init)
session.run(init_main)

for s in iter(range(max_epoch)):

    _start_time = time.time()

    batch_x = stories[start_index:end_index]
    batch_q = questions[start_index:end_index]
    batch_a = answers[start_index:end_index]

    fetch = [global_step, rnmodel.accuracy, rnmodel.loss, optimizer]

    fetch_values = session.run(
        fetch,
        feed_dict={
            rnmodel.x: batch_x,
            rnmodel.answers: batch_a,
            rnmodel.questions: batch_q,
            rnmodel.is_train: True,
            rnmodel.seq_length: seq_lens[start_index:end_index],
            rnmodel.seq_length_qs: questions_len[start_index:end_index],
            rnmodel.object_mask: object_mask[start_index:end_index],
            rnmodel.batch_size: int(len(batch_a))
        }
    )

    [step, accuracy, loss] = fetch_values[:3]

    step_time = time.time() - _start_time
    if s % 1 == 0:
        # periodic inference
        accuracy_test = session.run(
            rnmodel.accuracy, feed_dict={rnmodel.x: tstories,
                                         rnmodel.answers: tanswers,
                                         rnmodel.questions: tquestions,
                                         rnmodel.is_train: False,
                                         rnmodel.seq_length: tseq_lens,
                                         rnmodel.seq_length_qs: tquestions_len,
                                         rnmodel.object_mask: tobject_mask,
                                         rnmodel.batch_size: len(tanswers)})
        log_step_message(s, accuracy, accuracy_test, loss, step_time, False)

    start_index = end_index % total_examples
    end_index = min(start_index + batch_size, total_examples) % total_examples

# test the model
print("Testing Starts!")

accuracy_test = session.run(
            rnmodel.accuracy, feed_dict={rnmodel.x: tstories,
                                         rnmodel.answers: tanswers,
                                         rnmodel.questions: tquestions,
                                         rnmodel.is_train: False,
                                         rnmodel.seq_length: tseq_lens,
                                         rnmodel.seq_length_qs: tquestions_len,
                                         rnmodel.object_mask: tobject_mask,
                                         rnmodel.batch_size: len(tanswers)})
print("accuracy on entire test data is :  " + str(accuracy_test))