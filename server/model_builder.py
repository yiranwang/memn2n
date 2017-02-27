"""
==================================================================
This module defines and builds the model for training and testing.
1. build_memnn() builds the original babi_memnn model at: https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py
2. build_memnn_s() builds the model where the last merge is done with "sum" instead of "concat".
==================================================================
"""
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Permute, Dropout, LSTM, Input, Lambda, RepeatVector, merge # find Merge() classs in engine/topology.py



class MemNN(object):
    """
    This implements a multiple-hop End-To-End Memory Network.
    """
    def __init__(self, story_maxlen, query_maxlen, vocab_size, embedding_output_dim, hops = 3, weights_sharing = 'layerwise'):
        print "NOW IN MemNN() -----------------------------------------------"
        self.weights_sharing = weights_sharing
        self.hops = hops
        self.story_maxlen = story_maxlen
        self.query_maxlen = query_maxlen
        self.vocab_size = vocab_size
        self.embedding_output_dim = embedding_output_dim

    def _init_input(self):
        pass

    def _embedding_in(self, content = 'story'):
        """
        It returns an Embedding layer, callable to input tensor;
            embed the input tensor (S, ) to (V, D) space; (each int in input tensor corresponds to an vector of length D in (V, D) space)
            output a tensor (S, D)
        """
        if content == 'story':
            return Embedding(input_dim=self.vocab_size, # V
                             output_dim=self.embedding_output_dim, # D
                             input_length=self.story_maxlen) # S
        elif content == 'query':
            return Embedding(input_dim=self.vocab_size, # V
                             output_dim=self.embedding_output_dim, # D
                             input_length=self.query_maxlen) # Q
        else:
            raise Exception("ERROR in _embedding_in(): kwargs 'content' has to be either 'story' or 'query'.")

    def _embedding_out(self, content = 'story'):
        """
        It returns an Embedding layer, callable to input tensor;
            embed the input tensor (S, ) to (V, Q) space, where Q is the maxlength of queries.
            output a tensor (S, Q).
        """
        if content == 'story':
            return Embedding(input_dim=self.vocab_size,
                             output_dim=self.query_maxlen,
                             input_length=self.story_maxlen)
        elif content == 'query':
            return Embedding(input_dim=self.vocab_size,
                             output_dim=self.story_maxlen,
                             input_length=self.query_maxlen)
        else:
            raise Exception("ERROR in _embedding_out(): kwargs 'content' has to be either 'story' or 'query'.")

    def _match(self, story_encoder_in, query_encoder_in):
        """
        It computes the dot product of story_encoder_in tensor (S, D) and query_encoder_in tensor (Q, D), a "match" tensor (S, Q).
            and softened probability distribution after softmax(match) activation
            sums over each dimension in Q, got (S, )
            prepare it for matrix multiplication so we need to expand the sumed value to (S, Q) again.
        """
        match = merge([story_encoder_in, query_encoder_in], mode="dot", dot_axes=[2, 2]) # the 0-axs is for batch_size
        print "\t\t\t match shape:", match.get_shape()
        match = Activation("softmax")(match)
        print "\t\t\t match shape:", match.get_shape()
        match = Lambda(lambda x: K.sum(x, axis = 2))(match)
        print "\t\t\t match shape:", match.get_shape()
        match = RepeatVector(self.query_maxlen)(match)
        print "\t\t\t match shape:", match.get_shape()
        match = Permute((2, 1))(match)
        print "\t\t\t match shape:", match.get_shape()
        return match

    def _response(self, match, story_encoder_out):
        """
        It computes the element-wise multiplication of match tensor (S, Q) and story_encoder_out tensor (S, Q), a "response" tensor (S, Q)
            sigma the (Q, S) into a vector (Q, ) for next-stage use
            a merge via "sum" needs to be done in the next-stage self._sum_output() between (Q, ) and (Q, D) so..
            repeats the vector (Q, ) for D times to turn (Q, ) into (D, Q) and transposes it into (Q, D).
            NOW READY TO GO.
        """
        response = merge([match, story_encoder_out], mode="mul")
        print "\t\t\t response shape:", response.get_shape()
        response = Lambda(lambda x: K.sum(x, axis = 1))(response)
        print "\t\t\t response shape:", response.get_shape()
        response = RepeatVector(self.embedding_output_dim)(response)
        print "\t\t\t response shape:", response.get_shape()
        response = Permute((2, 1))(response)  # 0-axis: batch_size; 1-axis: query_maxlen; 2-axis: story_maxlen;
        print "\t\t\t response shape:", response.get_shape()

        return response

    def _sum_output(self, response, query_encoder_in):
        """
        It computes the sum of response (Q, D) and query_encider_in (Q, D)
            return a "out" tensor (Q, D)
        """
        out = merge([response, query_encoder_in], mode="sum")
        print "\t\t\t out shape:", out.get_shape()
        return out

    def _oneHop(self, story_encoder_in, query_encoder_in, story_encoder_out):
        match = self._match(story_encoder_in, query_encoder_in)
        print "\t\t match:", match.get_shape()
        response = self._response(match, story_encoder_out)
        print "\t\t response:", response.get_shape()
        sum_out = self._sum_output(response, query_encoder_in)
        print "\t\t sum_out:", sum_out.get_shape()
        return sum_out

    def _add_fc_layer(self, sum_out):
        """
        It computes the linear transformation of "out" tensor (Q, D) out of multiple hops into (V, ), followed by a sfmx activation
            returns a prediction probability distribution over vocabulary of size V
        """
        answer_output = Dense(self.vocab_size)(sum_out)
        print "\t\t answer_output:", answer_output.get_shape()
        answer_output = Activation("softmax")(answer_output)
        print "\t\t answer_output:", answer_output.get_shape()
        answer_output = Lambda(lambda x: K.sum(x, axis = 1))(answer_output)

        return answer_output


    def _build_layerwise_model(self):
        story_input = Input(shape=(self.story_maxlen,), dtype="int32")
        print "\t story_input:", story_input
        query_input = Input(shape=(self.query_maxlen,), dtype="int32")
        print "\t query_input:", query_input

        story_encoder_in = self._embedding_in(content = 'story')(story_input)
        print "\t story_encoder_in:", story_encoder_in.get_shape()
        story_encoder_out = self._embedding_out(content = 'story')(story_input)
        print "\t story_encoder_out:", story_encoder_out.get_shape()
        query_encoder_in = self._embedding_in(content = 'query')(query_input)
        print "\t query_encoder_in:", query_encoder_in.get_shape()

        last_out = query_encoder_in

        for _ in range(self.hops):

            last_out = self._oneHop(story_encoder_in, last_out, story_encoder_out)
            print "\t last_out:", last_out.get_shape()

        answer_output = self._add_fc_layer(last_out)
        print "answer_output:", answer_output.get_shape()

        model = Model(input=[story_input, query_input], output=[answer_output])
        return model


    def _build_dev_model(self):
        story_input = Input(shape=(self.story_maxlen,), dtype="int32")
        print "\t story_input:", story_input
        query_input = Input(shape=(self.query_maxlen,), dtype="int32")
        print "\t query_input:", query_input

        story_encoder_in = self._embedding_in(content = 'story')(story_input)
        print "\t story_encoder_in:", story_encoder_in.get_shape()
        story_encoder_out = self._embedding_out(content = 'story')(story_input)
        print "\t story_encoder_out:", story_encoder_out.get_shape()
        query_encoder_in = self._embedding_in(content = 'query')(query_input)
        print "\t query_encoder_in:", query_encoder_in.get_shape()

        last_out = self._oneHop(story_encoder_in, query_encoder_in, story_encoder_out)
        print "\t last_out:", last_out.get_shape()

        # print K.is_keras_tensor(last_out)
        #
        # last_out = Lambda(lambda x: K.floor(x))(last_out)
        # print K.is_keras_tensor(last_out)

        # query_encoder_in = self._embedding_in(content = 'query')(last_out)
        # print "\t query_encoder_in:", query_encoder_in.get_shape()

        last_out = self._oneHop(story_encoder_in, last_out, story_encoder_out)
        last_out = Lambda(lambda x: K.sum(x, axis = 2))(last_out)
        # match = self._match(story_encoder_in, query_encoder_in)
        # match (S, Q)

        # response = self._response(match, story_encoder_out)
        # response (Q, S)

        # answer_output = LSTM(4)(match)
        # answer_output = Lambda(lambda x: K.sum(x, axis = 2))(match)
        # print answer_output.get_shape()

        # answer_output = self._add_fc_layer(answer_output)
        answer_output = Dense(self.vocab_size)(last_out)
        print "\t\t answer_output:", answer_output.get_shape()


        return Model(input=[story_input, query_input], output=[answer_output])

    def build(self):
        if self.weights_sharing == 'layerwise':
            return self._build_layerwise_model()

        elif self.weights_sharing == 'dev':
            return self._build_dev_model()
        else:
            raise Exception("under dev...")


''' ----------------------------------------------------------------------------

def build_memnn(story_maxlen, query_maxlen, vocab_size, embedding_output_dim=64):

    print "NOW IN build_memnn() -----------------------------------------------"
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
    print "story_encoder_m in build_memnn() is in shape::", story_encoder_m.get_shape()
    # A Dropout() object is a layer, callable on a tensor, returns a tensor
    story_encoder_m = Dropout(0.3)(story_encoder_m)

    # same as embedding story_encoder_m
    # output: (batch_size, Q, D) where Q is the maxlength of queries.
    question_input = Input(shape=(query_maxlen,), dtype="int32")
    question_encoder = Embedding(input_dim=vocab_size,
                                 output_dim=embedding_output_dim,
                                 input_length=query_maxlen)(question_input)
    question_encoder = Dropout(0.3)(question_encoder)
    print "question_encoder in build_memnn() is in shape::", question_encoder.get_shape()
    # story_encoder_m: (batch_size, S, D), question_encoder: (batch_size, Q, D)
    # compute a 'match', a dot product between embeded story and embeded question.
    # output: (sabatch_sizemples, S, Q) go through sfmx activation interpreted as a probability vector over the input.
    match = merge([story_encoder_m, question_encoder], mode="dot",
                  dot_axes=[2, 2])
    match = Activation("softmax")(match)
    print "match in build_memnn() is in shape::", match.get_shape()
    # Notice the output_dim = Q instead of D
    # output: (batch_size, S, Q)
    story_encoder_c = Embedding(input_dim=vocab_size,
                                output_dim=query_maxlen,
                                input_length=story_maxlen)(story_input)
    story_encoder_c = Dropout(0.3)(story_encoder_c)
    print "story_encoder_c in build_memnn() is in shape::", story_encoder_c.get_shape()
    # response vector is the embeded input weighted by the "match" probability
    # output: permute (batch_size, S, Q) to (batch_size, Q, S)
    response = merge([match, story_encoder_c], mode="mul") # the mode used to be "sum"
    response = Permute((2, 1))(response)
    print "response in build_memnn() is in shape::", response.get_shape()

    # response: (batch_size, Q, S), question_encoder: (batch_size, Q, D)
    answer_encoder = merge([response, question_encoder], mode="concat", concat_axis=-1)
    print "right after concat -> answer_encoder in build_memnn() is in shape::", answer_encoder.get_shape()


    """
        The input to LSTM is in shape(batch_size, Q, S) corresponding to the classic format (sample_number, timesteps, input_dim)
        The output of LSTM is in shape(batch_size, Q, 32) where 32 is arbitrary number of hidden states.
    """
    answer_encoder = LSTM(32)(answer_encoder)
    print "right after LSTM -> answer_encoder in build_memnn() is in shape::", answer_encoder.get_shape()
    # one regularization layer -- more would probably be needed.
    answer_encoder = Dropout(0.3)(answer_encoder)
    answer_encoder = Dense(vocab_size)(answer_encoder)
    answer_output = Activation("softmax")(answer_encoder)
    # we output a probability distribution over the vocabulary
    print "answer_output in build_memnn() is in shape::", answer_output.get_shape()
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
--------------------------------------------------------------------------------
'''
