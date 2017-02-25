"""
==================================================================
This module:
1. trains End-To-End Memory Network built in model_builder.py on the datasets corresponding to challenges defined in qaTask.py
2. stores the trained model at the path defined in config.py
==================================================================
"""

from __future__ import print_function
from qaTasks import challenges
from training_data_utils import get_stories, vectorize_stories, get_babi_training_data, word2int_encode
from functools import reduce
import pickle
from config import OUTPUT_PATH, MODEL, NUM_EPOCHS
from model_builder import MemNN


def train_memnn_on(challenge_type, output_path, num_epochs):
    # Input: challenge_type = 'single_supporting_fact_10k'
    #        vocab_output_path = './server/model/babi_memnn/'
    #        num_epochs: number of epochs


    tar = get_babi_training_data() # origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'

    challenge = challenges[challenge_type]

    print('Extracting stories for the challenge:', challenge_type)
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    # with open('./server/model/babi_memnn/maxLength.txt', 'w'):


    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', story_maxlen, 'words')
    print('Query max length:', query_maxlen, 'words')
    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    print('-')
    print('Here\'s what a "story" tuple looks like (input, query, answer):')
    print(train_stories[0])
    print('-')
    print('Vectorizing the word sequences...')

    word_idx = word2int_encode(vocab)

    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    print('-')
    print('inputs: integer tensor of shape (samples, max_length)')
    print('inputs_train shape:', inputs_train.shape)
    print('inputs_test shape:', inputs_test.shape)
    print('-')
    print('queries: integer tensor of shape (samples, max_length)')
    print('queries_train shape:', queries_train.shape)
    print('queries_test shape:', queries_test.shape)
    print('-')
    print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
    print('answers_train shape:', answers_train.shape)
    print('answers_test shape:', answers_test.shape)
    print('-')
    print('Compiling...')

    # answer = MODEL(story_maxlen, query_maxlen, vocab_size) # -> "concat"

    # answer = MODEL(story_maxlen, query_maxlen, vocab_size, story_maxlen) # -> "sum"

    answer = MemNN(story_maxlen, query_maxlen, vocab_size, 64, hops = 3, weights_sharing = 'layerwise').build()

    answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

    answer.fit([inputs_train, queries_train], [answers_train],
               batch_size=32,
               nb_epoch=num_epochs,
               validation_data=([inputs_test, queries_test], [answers_test]))

    vocab_output_file = output_path + 'vocab_in_{}.pickle'.format(challenge_type)
    print('Saving vocab to {} ...'.format(output_path))
    with open(vocab_output_file, 'wb') as vocab_storage:
        pickle.dump(vocab, vocab_storage)
    print('Done!')

    print('Saving model to {} ...'.format(output_path))
    model_output_file = output_path + 'memnn_for_{}.h5'.format(challenge_type)
    answer.save(model_output_file)
    print('Done!')

    return story_maxlen, query_maxlen


def main(output_path, num_epochs):
    maxLength = {}
    for challenge_type in challenges:
        story_maxlen, query_maxlen = train_memnn_on(challenge_type, output_path, num_epochs)
        maxLength[challenge_type] = (story_maxlen, query_maxlen)

    with open(output_path + 'maxLength.pickle', 'wb') as maxlen_storage:
        pickle.dump(maxLength, maxlen_storage)

    return ;



if __name__ == '__main__':
    main(OUTPUT_PATH, NUM_EPOCHS)

################################################################
################################################################
################################################################
########  Cusion   #############################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################
