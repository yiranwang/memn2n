"""
==================================================================
This module helps to set up:
==================================================================
"""

import model_builder


'''
the model used during training.
'''
MODEL = model_builder.MemNN


'''
scheme kwargs:
    'hops':         bagOfWord model, word embedding - multiple hops - output
    'hops+lstm':    bagOfWord + sequential model, word embedding - multiple hops (sum) - LSTM - output
    'mem+lstm':     memory + sequential model, word embedding - one memory hop (concat) - LSTM - output
    'dev':          memory + sequential model, word embedding - LSTM encoding + multiple hops - output
'''
SCHEME = 'mem+lstm'
PARAMS = {'hops': 1, 'fixed_embedding': True}


'''
the path for outputing trained model and restoring the model
'''
OUTPUT_PATH = "./server/model/babi_memnn_1_1/"

'''
the number of training epochs
'''
NUM_EPOCHS = 80

'''
the name of the challenge type specified during demo.
'''
CHALLENGE_TYPE = "single_supporting_fact_10k"
