import os

from copy import deepcopy
import  numpy as np

sen_words = set()
def load_data(file_name):
    """ Load  Dataset.
    :param file_name
    :return: array of dict
    """
    tasks = []
    curr_task = None



    max_seq_len = 0
    max_ques_len = 0
    max_story_size = 0

    for i, line in enumerate(open(file_name)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            skip = False
            curr_task = {"Story": [], "Question": "", "Answer": ""}
            cur_story_size = 0

        line = line.strip()
        line = line.replace('.', ' .')
        line = line.replace('?', ' ?')
        line = line[line.find(' ') + 1:]
        if line.find('?') == -1:
            words = line.split(' ')
            sen_words.update(words)
            if max_seq_len < len(words):
                max_seq_len = len(words)
            curr_task["Story"].append(words)
            cur_story_size = cur_story_size +1
        else:
            idx = line.find('?')
            tmp = line[idx + 1:].split('\t')

            q_words = line[:idx+1].split(' ')
            sen_words.update(q_words)
            if max_ques_len < len(q_words):
                max_ques_len = len(q_words)
            if max_story_size < cur_story_size:
                max_story_size = cur_story_size
            curr_task["Question"] = q_words
            curr_task["Answer"] = tmp[1].strip()
            sen_words.add(curr_task["Answer"])

            tasks.append(deepcopy(curr_task))

    print("Loaded {} data from {}".format(len(tasks), file_name))

    return  tasks, max_story_size, max_seq_len, max_ques_len, len(sen_words)


def process_data(train,test):
    tasks1, max_story_size1, max_seq_len1, max_ques_len1, dim1 = load_data(train)
    tasks2, max_story_size2, max_seq_len2, max_ques_len2, dim2 = load_data(test)

    max_story_size = max(max_story_size1,max_story_size2)
    max_seq_len = max(max_seq_len1,max_seq_len2)
    max_ques_len = max(max_ques_len1,max_ques_len2)
    dim = max(dim1,dim2)
    tasks = tasks1+tasks2
    n = len(tasks)

    stories = []
    questions = []
    answers = []
    seq_lens = []
    q_lens = []
    V = {}
    index = 0
    for example in tasks:
        story, ques, answer, seq_len, q_len, V, index = convert(example, max_story_size, max_seq_len,
                                                                max_ques_len, dim, V, index)
        stories.append(story)
        questions.append(ques)
        answers.append(answer)
        seq_lens.append(seq_len)
        q_lens.append(q_len)

    stories = np.array(stories)
    questions = np.array(questions)
    answers = np.array(answers)
    seq_lens = np.array(seq_lens)
    q_lens = np.array(q_lens)

    object_mask = np.zeros([n, max_story_size, max_story_size])

    for i in range(n):
        for j in range(len(tasks[i]['Story'])):
            for k in range(len(tasks[i]['Story'])):
                object_mask[i][j][k] = 1

    print("Story: {} , Questions: {}, answers: {} ".format(stories.shape, questions.shape, answers.shape))
    return stories[0:len(tasks1)], stories[len(tasks1):],\
           questions[0:len(tasks1)], questions[len(tasks1):],\
           answers[0:len(tasks1)], answers[len(tasks1):],\
           seq_lens[0:len(tasks1)], seq_lens[len(tasks1):],\
           q_lens[0:len(tasks1)], q_lens[len(tasks1):],\
           object_mask[0:len(tasks1)],object_mask[len(tasks1):]


def convert(task, max_story_size, max_seq_len, max_ques_len, dim, V, index):
    # returns
    # story,
    # ques,
    # answer,
    # seq_len,
    # q_len

    seq_len = np.zeros([max_story_size])
    q_len = len(task['Question'])
    val, index = wordForIndex(V, index, task['Answer'], dim)
    answer = val

    ques = []
    for i in range(max_ques_len):
        if i < q_len:
            val, index = wordForIndex(V, index, task['Question'][i], dim)
            ques.append(val)
        else:
            ques.append(np.zeros([dim]))

    story = []
    story_len = len(task['Story'])
    for i in range(max_story_size):
        if i < story_len:
            sen_emb = []
            seq_len[i] = len(task['Story'][i])
            for j in range(max_seq_len):
                if j < seq_len[i]:
                    val, index = wordForIndex(V, index, task['Story'][i][j], dim)
                    sen_emb.append(val)
                else:
                    sen_emb.append(np.zeros([dim]))
            story.append(sen_emb)
        else:
            story.append(np.zeros([max_seq_len, dim]))

    return np.array(story), np.array(ques), answer, seq_len, q_len, V, index


def wordForIndex(V,index, word, dim):
    if word in V:
        return  V[word], index
    val = np.zeros([dim])
    val[index] = 1
    V[word] = val
    return val, index+1


#process_data('task21_blocksWorld_training.txt')
