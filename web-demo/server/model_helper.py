from memn2n import MemN2N
import tensorflow as tf
import numpy as np
import os


"""
Stores the configuration for loading the model
max_memory_size is the maximum size allowed during training
memory_size is the actual memory size used based on max story size
"""
config = {
    'batch': 64,
    'vocab_size': 20,
    'sentence_size': 6,
    'max_memory_size': 50,
    'memory_size': 10,
    'embedding_size': 40,
    'hops': 3,
    'max_grad_norm': 40.0,
    'regularization': 0.02,
    'epsilon': 1e-8,
    'lr': 0.01
}

def get_wt_dir_name():
    lr = config["lr"]
    eps = config["epsilon"]
    mgn = config["max_grad_norm"]
    hp = config["hops"]
    es = config["embedding_size"]
    ms = config["max_memory_size"]
    reg = config["regularization"]

    log_dir_name = "lr={0}_eps={1}_mgn={2}_hp={3}_es={4}_ms={5}_reg={6}".format(lr, eps, mgn, hp, es, ms, reg)
    return os.path.join('server/model/weights', log_dir_name)


restore_location = get_wt_dir_name()
sess = tf.Session()

model = MemN2N(config["batch"],
               config["vocab_size"],
               config["sentence_size"],
               config["memory_size"],
               config["embedding_size"],
               session=sess,
               hops=config["hops"],
               max_grad_norm=config["max_grad_norm"],
               l2=config["regularization"],
               lr=config["lr"],
               epsilon=config["epsilon"],
               nonlin=tf.nn.relu,
               restoreLoc=restore_location)


# Uncomment to see if the weights were loaded correctly
# print(sess.run(model.A))

def get_pred(testS, testQ):
    ps = model.predict_proba(testS, testQ)
    op = model.predict_test(testS, testQ)

    answer = op[0][0]
    answer_probability = float(np.max(ps))
    mem_probs = np.vstack(op[1:]).T[testS[0].any(axis=1)]
    return answer, answer_probability, mem_probs
