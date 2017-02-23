'''Trains a memory network on the bAbI dataset.
References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895
Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

from __future__ import print_function
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Dropout, Input, merge
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

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

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
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

# An Input() object is a tensor
story_input = Input(shape=(story_maxlen,), dtype="int32")

# An Embedding() object is a layer, callable on a tensor, returns a tensor
story_encoder_m = Embedding(input_dim=vocab_size,
                            output_dim=64,
                            input_length=story_maxlen)(story_input)

# A Dropout() object is a layer, callable on a tensor, returns a tensor
story_encoder_m = Dropout(0.3)(story_encoder_m)
# output: (samples, story_maxlen, embedding_dim)

# embed the question into a sequence of vectors
question_input = Input(shape=(query_maxlen,), dtype="int32")
question_encoder = Embedding(input_dim=vocab_size,
                             output_dim=64,
                             input_length=query_maxlen)(question_input)
question_encoder = Dropout(0.3)(question_encoder)
# output: (samples, query_maxlen, embedding_dim)

# compute a 'match' between input sequence elements (which are vectors)
# and the question vector sequence
match = merge([story_encoder_m, question_encoder], mode="dot",
              dot_axes=[2, 2])
# output: (samples, story_maxlen, query_maxlen)

# embed the input into a single vector with size = story_maxlen:
story_encoder_c = Embedding(input_dim=vocab_size,
                            output_dim=query_maxlen,
                            input_length=story_maxlen)(story_input)
story_encoder_c = Dropout(0.3)(story_encoder_c)
# output: (samples, story_maxlen, query_maxlen)

# sum the match vector with the input vector:
response = merge([match, story_encoder_c], mode="sum")
response = Permute((2, 1))(response)
# output: (samples, story_maxlen, query_maxlen)

# concatenate the match vector with the question vector,
# and do logistic regression on top
answer_encoder = merge([response, question_encoder], mode="concat", concat_axis=-1)
# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer_encoder = LSTM(32)(answer_encoder)
# one regularization layer -- more would probably be needed.
answer_encoder = Dropout(0.3)(answer_encoder)
answer_encoder = Dense(vocab_size)(answer_encoder)
answer_output = Activation("softmax")(answer_encoder)
# we output a probability distribution over the vocabulary
answer = Model(input=[story_input, question_input], output=[answer_output])

# compile network
answer.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# Using the functional API to avoid repeat the input twice
answer.fit([inputs_train, queries_train], [answers_train],
           batch_size=32,
           nb_epoch=1,
           validation_data=([inputs_test, queries_test], [answers_test]))




print(answer.layers)
print([layer.name.encode('utf8') for layer in answer.layers])

# later...
print ("saving model...")
json_file = open('model.json', 'w')
json_file.write(answer.to_json())
json_file.close()

print(answer.layers)
print([layer.name.encode('utf8') for layer in answer.layers])

print ("saving weights...")
answer.save_weights('model.h5')

print(answer.layers)
print([layer.name.encode('utf8') for layer in answer.layers])

# load json and create model
print ("loading model from json...")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

print(loaded_model.layers)
print([layer.name.encode('utf8') for layer in loaded_model.layers])

# load weights into new model
print ("loading weights from h5...")
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

print(loaded_model.layers)
print([layer.name.encode('utf8') for layer in loaded_model.layers])

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

# save model for loading later
answer.save_weights("/tmp/babi_memnn_copy.h5", overwrite=True)
config = answer.to_json()
with open("/tmp/babi_memnn_copy.json", "wb") as fjson:
    fjson.write(answer.to_json())
