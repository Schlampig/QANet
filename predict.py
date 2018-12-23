# -*- coding: utf-8 -*-

import os
import csv
import codecs
import json as json
import numpy as np
import tensorflow as tf
# load model directory
from config import Config
from model import Model
from preprocess import preprocess


# initialize graph and config
graph = tf.Graph()
config = Config()

# load dictionary
with open(config.word_dictionary, "r") as fh:
    word_dictionary = json.load(fh)
with open(config.char_dictionary, "r") as fh:
    char_dictionary = json.load(fh)
with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
with open(config.char_emb_file, "r") as fh:
    char_mat = np.array(json.load(fh), dtype=np.float32)

# define model structre
model = Model(config, batch=None, word_mat=word_mat, char_mat=char_mat, trainable=False, demo=True, graph=graph)

# start session
with graph.as_default() as g:
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

    
def predict(q, c):
    # input: q (question) and c (context) are two strings
    # ouput: ans (answer) is a string
    global model, config, word_dictionary, char_dictionary
    # transform strings to indexes
    context, context_char = preprocess(c, config, word_dictionary, char_dictionary, query_type='context')
    question, question_char = preprocess(q, config, word_dictionary, char_dictionary, query_type='question')
    # predict
    if config.use_char_emb:
        fd = {'context:0': [context], 
              'question:0': [question], 
              'context_char:0': [context_char],
              'question_char:0': [question_char]}
    else:
        fd = {'context:0':[context], 
              'question:0':[question]}
    yp1, yp2 = sess.run([model.yp1, model.yp2], feed_dict=fd)
    ans = "".join(c[yp1[0]:yp2[0]])
    return ans
    

if __name__ == '__main__':
    q = "佩鲁贾被什么队击败？"
    c = """佛罗伦萨曾于五十至六十年代两夺意甲联赛冠军，之后长期处于联赛中下游，并多次降班。2002年因财赤宣布破产，被意大利赛会判罚降班。之后得到鞋业商人德拉瓦莱（Diego Della Valle）的支持组成新球队 Florentia Viola，并于意大利丙二组联赛开始比赛。2003年夏天，德拉瓦莱买回“Fiorentina”这个名称，并再次以佛罗伦萨的名称参加比赛。2004年在升级附加赛中击败佩鲁贾，完成了令人惊艳的三级跳，顺利重返意甲联赛。"""
    print(predict(q, c))
    
