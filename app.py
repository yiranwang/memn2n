"""
==================================================================
This module:
1. reload the trained module corresponding to the given challenge types defined in config.py
2. serves up the web service and listens to the user input
==================================================================
"""

from flask import Flask, request, jsonify
import server.testing_data_utils
import server.babi_memnn_testing
import server.training_data_utils
from server.config import OUTPUT_PATH, CHALLENGE_TYPE
import numpy as np
import json
from server.config import MODEL, SCHEME


# prepare task specific vocabulary and parametes
vocab, vocab_size, story_maxlen, query_maxlen = server.testing_data_utils.load_testing_story_params(OUTPUT_PATH, CHALLENGE_TYPE)
word_idx = server.training_data_utils.word2int_encode(vocab)
idx_word = server.testing_data_utils.int2word_decode(word_idx)

# reload trained model
answer = server.babi_memnn_testing.recover_model_from(OUTPUT_PATH + '{}_{}.h5'.format(SCHEME, CHALLENGE_TYPE))

app = Flask(__name__, static_url_path='')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/answer', methods=['GET', 'POST'])
def get_answer():
    data = json.loads(request.form['data'])
    sentences = data['sentences']
    question = data['question']

    testS, testQ, testA = server.testing_data_utils.process_data(sentences, question, story_maxlen, query_maxlen, word_idx)
    # answer, answer_probability, mem_probs = get_pred(testS, testQ)
    answer_code, answer_confidence, prob_dist = server.babi_memnn_testing.get_pred(answer, [testS, testQ])

    response = {
        "answer": server.testing_data_utils.answer_decode(answer_code, idx_word),
        "answerProbability": answer_confidence,
        # "memoryProbabilities": prob_dist
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run()

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
