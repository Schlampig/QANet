import os
import re
import csv
import time
import string
import numpy as np
import ujson as json
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from flask import Flask, request

import jieba
from model import Model
from config import flags
from preprocess import preprocess


### Global Setting
app = Flask(__name__)
config = flags.FLAGS
graph = tf.Graph()
with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
with open(config.word_dictionary,'r') as f:
    word_dict = json.load(f)

model = Model(config, word_mat=word_mat, trainable=False, opt=False, demo=True, graph = graph)

default_q = [[1,2]]
default_c = [[3,4]]
with graph.as_default() as g:
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
    if config.decay < 1.0:
        sess.run(model.assign_vars)
    qa_id, logits, yp1, yp2 = sess.run([model.qa_id, model.logits, model.yp1, model.yp2], 
                                       feed_dict={model.c:default_c, model.q:default_q})

    
def readingComprehension(para, query):
    c, seg_c = preprocess(para, config, word_dict)
    q, _ = preprocess(query, config, word_dict)
    qa_id, logits, yp1, yp2 = sess.run([model.qa_id, model.logits, model.yp1, model.yp2], 
                                       feed_dict={model.c:c, model.q:q})
    answer = ''.join(seg_c[yp1[0]: yp2[0]+1])     
    return answer


@app.route("/", methods=["POST"])
def hello():
    json_str = request.json
    context, question = json_str.get("context"), json_str.get("question")
    result = readingComprehension(context, question)
    return str(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8098, threaded=True)
    
