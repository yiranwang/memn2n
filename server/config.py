"""
==================================================================
This module helps to set up:
==================================================================
"""

import model_builder


'''
the model used during training.
'''
MODEL = model_builder.build_memnn

'''
the path for outputing trained model and restoring the model
'''
OUTPUT_PATH = "./server/model/babi_memnn_1_1/"

'''
the number of training epochs
'''
NUM_EPOCHS = 1

'''
the name of the challenge type specified during demo.
'''
CHALLENGE_TYPE = "single_supporting_fact_10k"
