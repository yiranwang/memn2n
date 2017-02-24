"""
==================================================================
This module defines and builds the model for training and testing.
1. build_memnn() builds the original babi_memnn model at: https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
2. build_memnn_s() builds the model where the last merge is done with "sum" instead of "concat".
==================================================================
"""

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Dropout, LSTM, Input, merge # find Merge() classs in engine/topology.py


def build_memnn(story_maxlen, query_maxlen, vocab_size, embedding_output_dim=64):
    # An Input() object is a tensor
    story_input = Input(shape=(story_maxlen,), dtype="int32")

    # An Embedding() object is a layer, callable on a tensor
    # takes in an Input() object in shape(S, ) with S is the maxlength of stories,
    # embeds each s in space(V, D) where s is in vocabulary of size V.
    # returns a tensor in shape(S, D)
    # output: (batch_size, S, D)
    story_encoder_m = Embedding(input_dim=vocab_size, # V
                                output_dim=embedding_output_dim, # D
                                input_length=story_maxlen)(story_input) # S

    # A Dropout() object is a layer, callable on a tensor, returns a tensor
    story_encoder_m = Dropout(0.3)(story_encoder_m)

    # same as embedding story_encoder_m
    # output: (batch_size, Q, D) where Q is the maxlength of queries.
    question_input = Input(shape=(query_maxlen,), dtype="int32")
    question_encoder = Embedding(input_dim=vocab_size,
                                 output_dim=embedding_output_dim,
                                 input_length=query_maxlen)(question_input)
    question_encoder = Dropout(0.3)(question_encoder)

    # story_encoder_m: (batch_size, S, D), question_encoder: (batch_size, Q, D)
    # compute a 'match', a dot product between embeded story and embeded question.
    # output: (sabatch_sizemples, S, Q) go through sfmx activation interpreted as a probability vector over the input.
    match = merge([story_encoder_m, question_encoder], mode="dot",
                  dot_axes=[2, 2])
    match = Activation("softmax")(match)

    # Notice the output_dim = Q instead of D
    # output: (batch_size, S, Q)
    story_encoder_c = Embedding(input_dim=vocab_size,
                                output_dim=query_maxlen,
                                input_length=story_maxlen)(story_input)
    story_encoder_c = Dropout(0.3)(story_encoder_c)

    # response vector is the embeded input weighted by the "match" probability
    # output: permute (batch_size, S, Q) to (batch_size, Q, S)
    response = merge([match, story_encoder_c], mode="mul") # the mode used to be "sum"
    response = Permute((2, 1))(response)

    # response: (batch_size, Q, S), question_encoder: (batch_size, Q, D)
    answer_encoder = merge([response, question_encoder], mode="concat", concat_axis=-1)
    answer_encoder = LSTM(32)(answer_encoder)
    # one regularization layer -- more would probably be needed.
    answer_encoder = Dropout(0.3)(answer_encoder)
    answer_encoder = Dense(vocab_size)(answer_encoder)
    answer_output = Activation("softmax")(answer_encoder)
    # we output a probability distribution over the vocabulary
    answer = Model(input=[story_input, question_input], output=[answer_output])

    return answer


def build_memnn_s(story_maxlen, query_maxlen, vocab_size, embedding_output_dim):
    """
    If we want to merge response and question_encoder via "sum" instead of "concat",
    so that the dimension wouldn't increase as we stack more layers,
    we need to ensure embedding_output_dim == story_maxlen.
    """
    if story_maxlen != embedding_output_dim:
        raise Exception('ERROR in build_memnn_s(): model failed to build because story_maxlen NOT equals embedding_output_dim')

    # An Input() object is a tensor
    story_input = Input(shape=(story_maxlen,), dtype="int32")

    # An Embedding() object is a layer, callable on a tensor
    # takes in an Input() object in shape(S, ) with S is the maxlength of stories,
    # embeds each s in space(V, D) where s is in vocabulary of size V
    # ----------------------------------------------------------------------
    # in our case, the embedding space is (V, S) where the embedding_output_dim is S.
    # ----------------------------------------------------------------------
    # returns a tensor in shape(S, D)
    # output: (batch_size, S, D) -> corresponding to classic input dimensionality to LSTM gate (batch_size, sequence_of_input, each_input_length)
    story_encoder_m = Embedding(input_dim=vocab_size, # V
                                output_dim=embedding_output_dim, # D
                                input_length=story_maxlen)(story_input) # S

    # A Dropout() object is a layer, callable on a tensor, returns a tensor
    story_encoder_m = Dropout(0.3)(story_encoder_m)

    # same as embedding story_encoder_m
    # output: (batch_size, Q, D) where Q is the maxlength of queries.
    question_input = Input(shape=(query_maxlen,), dtype="int32")
    question_encoder = Embedding(input_dim=vocab_size,
                                 output_dim=embedding_output_dim,
                                 input_length=query_maxlen)(question_input)
    question_encoder = Dropout(0.3)(question_encoder)

    # story_encoder_m: (batch_size, S, D), question_encoder: (batch_size, Q, D)
    # compute a 'match', a dot product between embeded story and embeded question.
    # output: (batch_size, S, Q) go through sfmx activation interpreted as a probability vector over the input.
    match = merge([story_encoder_m, question_encoder], mode="dot", # story_encoder_m.dot(question_encoder.T)
                  dot_axes=[2, 2])
    match = Activation("softmax")(match)

    # Notice the only difference of this embedding layer: output_dim = Q instead of D
    # output: (batch_size, S, Q)
    story_encoder_c = Embedding(input_dim=vocab_size,
                                output_dim=query_maxlen,
                                input_length=story_maxlen)(story_input)
    story_encoder_c = Dropout(0.3)(story_encoder_c)


    # match: (batch_size, S, Q), story_encoder_c: (batch_size, S, Q)
    # response vector is the embeded input weighted by the "match" probability
    response = merge([match, story_encoder_c], mode="mul") # the mode used to be "sum"
    # output: permute (batch_size, S, Q) to (batch_size, Q, S)
    response = Permute((2, 1))(response)


    """
    response: (batch_size, Q, S), question_encoder: (batch_size, Q, D)
    remember we have D equal to S, we can choose mode = 'sum'
    """
    answer_encoder = merge([response, question_encoder], mode="sum")

    """
    The input to LSTM is in shape(batch_size, Q, S) corresponding to the classic format (sample_number, timesteps, input_dim)
    The output of LSTM is in shape(batch_size, Q, 32) where 32 is arbitrary number of hidden states.
    """
    answer_encoder = LSTM(32)(answer_encoder)
    # one regularization layer -- more would probably be needed.
    answer_encoder = Dropout(0.3)(answer_encoder)
    # W(response + question_encoder) outputs (batch_size, V) where V is the vocab_size
    answer_encoder = Dense(vocab_size)(answer_encoder)
    # sfmx(W(response + question_encoder)) outputs the probability distribution over the vocabulary
    answer_output = Activation("softmax")(answer_encoder)

    answer = Model(input=[story_input, question_input], output=[answer_output])

    return answer
