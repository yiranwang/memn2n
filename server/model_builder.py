"""
==================================================================
This module defines and builds the model for training and testing.
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
    scheme:
        'hops': bagOfWord model, word embedding - multiple hops - output
        'hops+lstm': bagOfWord + sequential model, word embedding - multiple hops (sum) - LSTM - output
        'mem+lstm': memory + sequential model, word embedding - one memory hop (concat) - LSTM - output
        'dev': memory + sequential model, word embedding - LSTM encoding + multiple hops - output
        ''
    """
    def __init__(self, story_maxlen, query_maxlen, vocab_size, embedding_output_dim, params, scheme = 'hops', debug = False):
        self.story_maxlen = story_maxlen
        self.query_maxlen = query_maxlen
        self.vocab_size = vocab_size
        self.embedding_output_dim = embedding_output_dim
        self.debug = debug
        self.scheme = scheme

        self.hops = params['hops']
        self.fixed_embedding = params['fixed_embedding']
        self.story_input = None
        self.query_input = None

        if self.scheme == 'mem+lstm' and self.hops != 1:
            raise Exception('"mem+lstm" only allows for 1 hop.')

        if self.debug:
            print "NOW IN MemNN() -----------------------------------------------"

    def _create_input(self):
        self.story_input = Input(shape=(self.story_maxlen,), dtype="float32")
        self.query_input = Input(shape=(self.query_maxlen,), dtype="float32")
        return ;

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
        match = Activation('softmax')(match)
        return match

    def _response(self, match, story_encoder_out, mode = 'sum'):
        """
        It computes the element-wise multiplication of match tensor (S, Q) and story_encoder_out tensor (S, Q), a "response" tensor (S, Q)
            sigma the (Q, S) into a vector (Q, ) for next-stage use
            a merge via "sum" needs to be done in the next-stage self._sum_output() between (Q, ) and (Q, D) so..
            repeats the vector (Q, ) for D times to turn (Q, ) into (D, Q) and transposes it into (Q, D).
            NOW READY TO GO.
        """
        response = merge([match, story_encoder_out], mode=mode)

        if self.scheme == 'hops' or self.scheme == 'dev':
            response = Lambda(lambda x: K.sum(x, axis = 1))(response)
        elif self.scheme == 'hops+lstm':
            response = Lambda(lambda x: K.sum(x, axis = 1))(response)
            response = RepeatVector(self.embedding_output_dim)(response)
            response = Permute((2, 1))(response)  # 0-axis: batch_size; 1-axis: query_maxlen; 2-axis: story_maxlen;
        else:
            response = Permute((2, 1))(response)  # 0-axis: batch_size; 1-axis: query_maxlen; 2-axis: story_maxlen;

        return response

    def _sum_output(self, response, query_encoder_in, mode = 'concat'):
        """
        It computes the sum of response (Q, S) and query_encider_in (Q, D)
            return a "out" tensor (Q, S + D)
        """
        out = merge([response, query_encoder_in], mode=mode)

        return out


    def _add_fc_layer(self, sum_out):
        """
        It computes the linear transformation of "out" tensor (Q, D) out of multiple hops into (V, ), followed by a sfmx activation
            returns a prediction probability distribution over vocabulary of size V
        """
        answer_output = Dense(self.vocab_size)(sum_out)

        answer_output = Activation("softmax")(answer_output)

        return answer_output

    def _hopEmbedding(self):
        """
        It helps embed story and query specifically for hops models.
        """
        if self.fixed_embedding is None or self.story_input is None or self.query_input is None:
            raise Exception("Error in self._hopEmbedding()")

        if self.fixed_embedding:
            story_encoder_in = self._embedding_in(content = 'story')(self.story_input)
            story_encoder_in = [story_encoder_in for _ in range(self.hops)]
            story_encoder_out = self._embedding_out(content = 'story')(self.story_input)
            story_encoder_out = [story_encoder_out for _ in range(self.hops)]
            query_encoder_in = self._embedding_in(content = 'query')(self.query_input)
            query_encoder_in = [query_encoder_in for _ in range(self.hops)]
        else:
            story_encoder_in = [self._embedding_in(content = 'story')(self.story_input) for _ in range(self.hops)]
            story_encoder_out = [self._embedding_out(content = 'story')(self.story_input) for _ in range(self.hops)]
            query_encoder_in = [self._embedding_in(content = 'query')(self.query_input) for _ in range(self.hops)]

        return story_encoder_in, story_encoder_out, query_encoder_in

    def _build_hops_model(self):
        """
        3 is a relatively good hop number.
        Using higher number of hops is equivalent to paying attentions to a lot of different parts of the story equally.
        Fundamentally, the model capacity quickly hit the roof this way.
        Fixed embeddings achieved similar results as variable embeddings,
            suggesting that 3-hop model helps improve the acurracy not because it increases the number of the weights but actually it allows the model to compare and shift attention.
        """
        self._create_input()

        story_encoder_in, story_encoder_out, query_encoder_in = self._hopEmbedding()

        last_out = self.query_input
        for i in range(self.hops):
            match = self._match(story_encoder_in[i], query_encoder_in[i])
            # (S, Q) = (S, D) dot (D, Q)
            response = self._response(match, story_encoder_out[i], mode = 'mul')
            # (Q, ) = (sum((S, Q) merge_mul (S, Q), axis = S)
            last_out = self._sum_output(response, last_out, mode = 'sum')
            # (Q, ) = (Q, ) sum_merge (Q, )

        sum_out = last_out
        # sum_out = Lambda(lambda x: K.sum(x, axis = 1))(last_out)
        # (D, ) = sum((Q, D), axis = Q)
        answer_output = self._add_fc_layer(sum_out)

        if self.debug:
            print "\t last_out shape:", last_out.get_shape()
            print "\t sum_out shape:", sum_out.get_shape()
            print "\t answer_output:", answer_output.get_shape()

        return Model(input=[self.story_input, self.query_input], output=[answer_output])


    def _build_hops_lstm_model(self):
        self._create_input()

        story_encoder_in, story_encoder_out, query_encoder_in = self._hopEmbedding()

        last_out = query_encoder_in[0]
        for i in range(self.hops):
            match = self._match(story_encoder_in[i], last_out)
            # (S, Q) = (S, D) dot (D, Q)
            response = self._response(match, story_encoder_out[i], mode = 'mul')
            # (Q, D) = permute(expand_to_D(sum((S, Q) merge_mul (S, Q), axis = S)))
            last_out = self._sum_output(response, query_encoder_in[i], mode = 'sum')
            # (Q, D) = (Q, D) merge_mul (Q, D)

        seq_out = LSTM(32)(last_out)

        answer_output = self._add_fc_layer(seq_out)

        if self.debug:
            print "\t last_out shape:", last_out.get_shape()
            print "\t answer_output:", answer_output.get_shape()

        return Model(input=[self.story_input, self.query_input], output=[answer_output])


    def _build_mem_lstm_model(self):
        """
        80%
        """
        self._create_input()

        story_encoder_in, story_encoder_out, query_encoder_in = self._hopEmbedding()

        match = self._match(story_encoder_in[0], query_encoder_in[0])
        response = self._response(match, story_encoder_out[0], mode = 'mul')
        sum_out = self._sum_output(response, query_encoder_in[0], mode = 'concat')

        answer_output = LSTM(32)(sum_out)
        answer_output = self._add_fc_layer(answer_output)

        return Model(input=[self.story_input, self.query_input], output=[answer_output])


    def _build_dev_model(self):
        self._create_input()

        story_encoder_in, story_encoder_out, query_encoder_in = self._hopEmbedding()

        last_out = self.query_input
        for i in range(self.hops):
            seq_story_encoder = LSTM(self.query_maxlen)(story_encoder_in[i])
            seq_query_encoder = LSTM(self.query_maxlen)(query_encoder_in[i])
            seq_story_query = merge([seq_story_encoder, seq_query_encoder], mode = 'sum')

            match = self._match(story_encoder_in[i], query_encoder_in[i])
            # (S, Q) = (S, D) dot (D, Q)
            response = self._response(match, story_encoder_out[i], mode = 'mul')
            # (Q, ) = sum((S, Q) merge_mul (S, Q), axis = S))
            last_out = self._sum_output(response, seq_story_query, mode = 'sum')
            # (Q, ) = (Q, ) merge_mul (Q, )

        # seq_out = LSTM(32)(last_out)
        answer_output = self._add_fc_layer(last_out)

        if self.debug:
            print "\t last_out shape:", last_out.get_shape()
            print "\t answer_output:", answer_output.get_shape()

        return Model(input=[self.story_input, self.query_input], output=[answer_output])


    def build(self):
        if self.scheme == "hops":
            return self._build_hops_model()
        if self.scheme == "hops+lstm":
            return self._build_hops_lstm_model()
        if self.scheme == 'mem+lstm':
            return self._build_mem_lstm_model()
        if self.scheme == "dev":
            return self._build_dev_model()
        else:
            raise Exception("under dev...")



        # if self.debug:
        #     print "\t story_input:", story_input.get_shape()
        #     print "\t query_input:", query_input.get_shape()
        #     print "\t story_encoder_in:", story_encoder_in.get_shape()
        #     print "\t story_encoder_out:", story_encoder_out.get_shape()
        #     print "\t query_encoder_in:", query_encoder_in.get_shape()
