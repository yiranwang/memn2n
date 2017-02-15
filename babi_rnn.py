'''Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.

The results are comparable to those for an LSTM model provided in Weston et al.:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698

Task Number                  | FB LSTM Baseline | Keras QA
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               | 100.0
QA2 - Two Supporting Facts   | 20               | 50.0
QA3 - Three Supporting Facts | 20               | 20.5
QA4 - Two Arg. Relations     | 61               | 62.9
QA5 - Three Arg. Relations   | 70               | 61.9
QA6 - Yes/No Questions       | 48               | 50.7
QA7 - Counting               | 49               | 78.9
QA8 - Lists/Sets             | 45               | 77.2
QA9 - Simple Negation        | 64               | 64.0
QA10 - Indefinite Knowledge  | 44               | 47.7
QA11 - Basic Coreference     | 72               | 74.9
QA12 - Conjunction           | 74               | 76.4
QA13 - Compound Coreference  | 94               | 94.4
QA14 - Time Reasoning        | 27               | 34.8
QA15 - Basic Deduction       | 21               | 32.4
QA16 - Basic Induction       | 23               | 50.6
QA17 - Positional Reasoning  | 51               | 49.1
QA18 - Size Reasoning        | 52               | 90.8
QA19 - Path Finding          | 8                | 9.0
QA20 - Agent's Motivations   | 91               | 90.7

For the resources related to the bAbI project, refer to:
https://research.facebook.com/researchers/1543934539189348

Notes:

- With default word, sentence, and query vector sizes, the GRU model achieves:
  - 100% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
  - 50% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)
In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.

- The task does not traditionally parse the question separately. This likely
improves accuracy and is a good example of merging two RNNs.

- The word vector embeddings are not shared between the story and question RNNs.

- See how the accuracy changes given 10,000 training samples (en-10k) instead
of only 1000. 1000 was used in order to be comparable to the original paper.

- Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.

- The length and noise (i.e. 'useless' story components) impact the ability for
LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
networks that use attentional processes can efficiently search through this
noise to find the relevant statements, improving performance substantially.
This becomes especially obvious on QA2 and QA3, both far longer than QA1.
'''

from __future__ import print_function
from functools import reduce
import re
import tarfile
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences



def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()] # regex of a word



def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    -------------------
    Parsing format:
    Each file contains a number of stories, which consists of a number of lines of content.
    Indice 1 indicates the 1st line of a story.
    Every two lines story telling are followed by a line of question + answer + indice of supporting line separated by '\t'.
    So every story has one or multiple questions answered.
    -------------------

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = [] # Beginning of a story, new a list.
        if '\t' in line:
            q, a, supporting = line.split('\t') #
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split()) # iterable = map(func. iterable), supporting could be more than one indice.
                substory = [story[i - 1] for i in supporting]
            else:
                # As story unfolds question by question, the next question corresponds to the current and all previous story lines.
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('') # To keep indices in story consistent, add an empty string instead when dealing with question line.
        else: # Add story line when not dealing with question line.
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting) # f.readlines() is used as iterface.

    # flatten() explained
    # -------------------
    # data above received as in [([[storyline1], [storyline2]], [question_line], answer), () ... ()] where story lines are grouped.
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    # flatten() takes in story and returns reduced outcome. reduce() apply "+" operation on one element and the next in story, which equals to list1.extend(list2)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    # data now received as in [([storyline1, storyline2], [question_line], answer), () ... ()] where story lines are flattened.
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story] # word_idx is the dictionary = {"word": indice}.
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1 # answer didn't go through tokenization, so it is a single word.
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    # Each x is a sequence of words, thus a 1D array.
    # X is a series of stories, thus a 2D array.
    # Pad each x, whose length is less than story_maxlen, with default value 0 of default type of int32, and return a numpy.array.
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 1
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)


# Choose your desired samples
# ----------------------------
# Default QA1 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

# Set training and testing sets
# ------------------------------
train = get_stories(tar.extractfile(challenge.format('train')))
test = get_stories(tar.extractfile(challenge.format('test')))

# lamda x, y: x | y, (set()) explained:
# -------------------------------------
# for story, q, answer in train + test:
#   set(story + q + [answer]) -> a set of vocabulary appeared in this unit, train + test contains a number of such units
# (set([s + q + a]) for s, q, a in iterator) -> compress all sets of vocabulary generated into a Generator as the third argument passed into reduce()
# reduce() unions("|") one set and the next to form a set of total vocabulary.
vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1

word_idx = dict((c, i + 1) for i, c in enumerate(vocab)) # Generator = ((c, i + 1) for i, c in enumerate(vocab)); Dictionary = dict(Generator)

story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')


"""
recurrent.Recurrent() is the base class used in recurrent.LSTM/recurrent.GRU

Only the 1st layer needs to specify input_dim and output_dim.
All the following layers require output_dim.
"""

sentrnn = Sequential()
sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                      input_length=story_maxlen))
sentrnn.add(RNN(SENT_HIDDEN_SIZE, return_sequences = True))
sentrnn.add(Dropout(0.3))
# return_sequences = True so that the output follows (sample_nb, story_maxlen, SENT_HIDDEN_SIZE)
# else the output follows (sample_nb, SENT_HIDDEN_SIZE)

qrnn = Sequential()
qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                   input_length=query_maxlen))

qrnn.add(RNN(QUERY_HIDDEN_SIZE, return_sequences = False))
qrnn.add(Dropout(0.3))
qrnn.add(RepeatVector(story_maxlen))
# If set return_sequences = True, the output follows (sample_nb, query_maxlen, QUERY_HIDDEN_SIZE)
# To match up with the output format of sentrnn, set return_sequences = False so that the output follows (sample_nb, QUERY_HIDDEN_SIZE)
# qrnn.add(RepeatVector(story_maxlen)) layer makes it to (sample_nb, story_maxlen, QUERY_HIDDEN_SIZE)

model = Sequential()
model.add(Merge([sentrnn, qrnn], mode='concat'))
model.add(RNN(SENT_HIDDEN_SIZE + QUERY_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

# everything above this line is training process modified based on the original keras repo
# ------------------------------------------------------------------------
# everything below this line is

import pickle
vocab_output_path = '/Users/shayangzang/Desktop/cs274c-deeplearning/QA_project/memn2n/web-demo/server/model/babi_rnn/v1.pickle'
with open(vocab_output_path, 'wb') as vocab_storage:
    pickle.dump(vocab, vocab_storage)

print('Saving vocab to {} ... Done!'.format(vocab_output_path))
model_output_path = '/Users/shayangzang/Desktop/cs274c-deeplearning/QA_project/memn2n/web-demo/server/model/babi_rnn/m1.h5'
model.save(model_output_path)
print('Saving model to {} ... Done!'.format(model_output_path))






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
