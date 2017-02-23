"""
This module defines functions that help:
1. load the vocabulary related story params corresponding to given challenge type during testing
2. convert user raw input data to standard format so it can be fed in the trained model during testing
"""


import pickle
from training_data_utils import tokenize, vectorize_stories


def load_testing_story_params(output_path, challenge_type):

    vocab_file = output_path + 'vocab_in_{}.pickle'.format(challenge_type)

    print "load vocab_file from:", vocab_file

    with open(vocab_file, 'rb') as handle:
        vocab = pickle.load(handle)

    print "vocab includes:"
    for word in vocab:
        print word

    vocab_size = len(vocab) + 1

    maxlength_file = output_path + 'maxLength.pickle'
    with open(maxlength_file, 'rb') as handle:
        maxLength = pickle.load(handle)
        story_maxlen, query_maxlen = maxLength[challenge_type]

    return vocab, vocab_size, story_maxlen, query_maxlen


"""
Input: answer_code in numpy.array format, whose length should be 1.
"""
def int2word_decode(word_idx):
    if not isinstance(word_idx, dict) or len(word_idx) <= 0:
        raise Exception("ERROR in decode(): input word_idx type/shape wrong.")
    idx_word = dict((v, k) for k, v in word_idx.iteritems())
    return idx_word

def answer_decode(answer_code, idx_word):
    return idx_word.get(answer_code, 'unknown')


"""
the main function called in app.py to deal with user input
"""
def process_data(sentences, question, story_maxlen, query_maxlen, word_idx):
    sent_t = [tokenize(s.lower()) for s in sentences]
    sent_t = [filter(lambda x: x != ".", s) for s in sent_t]

    q_t = tokenize(question.lower())

    data = [(sent_t, q_t, 'where')] # match the data format before going to data flatten
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not story_maxlen or len(flatten(story)) < story_maxlen] # flaten story lines

    testS, testQ, testA = vectorize_stories(data, word_idx, story_maxlen, query_maxlen)

    return testS, testQ, testA
