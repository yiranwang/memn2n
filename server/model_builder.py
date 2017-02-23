"""
==================================================================
This module defines and builds the model for training and testing.
==================================================================
"""

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Dropout, LSTM, Input, merge # find Merge() classs in engine/topology.py


def build_memnn(story_maxlen, query_maxlen, vocab_size, embedding_output_dim=64):
    # An Input() object is a tensor
    story_input = Input(shape=(story_maxlen,), dtype="int32")

    # An Embedding() object is a layer, callable on a tensor, returns a tensor
    story_encoder_m = Embedding(input_dim=vocab_size,
                                output_dim=embedding_output_dim,
                                input_length=story_maxlen)(story_input)

    # A Dropout() object is a layer, callable on a tensor, returns a tensor
    story_encoder_m = Dropout(0.3)(story_encoder_m)
    # output: (samples, story_maxlen, embedding_dim)

    # embed the question into a sequence of vectors
    question_input = Input(shape=(query_maxlen,), dtype="int32")
    question_encoder = Embedding(input_dim=vocab_size,
                                 output_dim=embedding_output_dim,
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

    return answer
