import os
import re
import csv
import time
import string
import numpy as np
import json as json
from tqdm import tqdm
import tensorflow as tf
from collections import Counter

import jieba
from model import Model
from config import flags
from preprocess import preprocess
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset


# Global Settings
config = flags.FLAGS
graph = tf.Graph()
with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
with open('data/word_dictionary.json', "r") as fh:
    word_dict = json.load(fh)

model = Model(config, word_mat=word_mat, trainable=False, opt=False, demo=True, graph=graph)

default_q = [[1, 2]]
default_c = [[3, 4]]
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


def search_tagging(para, query):
    c, seg_c = preprocess(para, config, word_dict)
    q, _ = preprocess(query, config, word_dict)
    qa_id, logits, yp1, yp2 = sess.run([model.qa_id, model.logits, model.yp1, model.yp2], 
                                       feed_dict={model.c:c, model.q:q})
    answer = ''.join(seg_c[yp1[0]: yp2[0]+1])
    return answer


if __name__ == '__main__':
    c = """武磊，1991年11月19日出生于中国南京，中国足球运动员，司职前锋，效力于西甲皇家西班牙人俱乐部。2004 年，武磊在亚足联U14少年足球分区赛上打入6球助中国队力压日本夺冠。2006年，14岁10个月的武磊在中乙联赛登场，成为中国职业足球赛场史上最年轻的球员。2012年，武磊单赛季17个入球，以中甲本土射手王的身份带领上海东亚冲超成功。2013年在中国对澳大利亚的比赛中打入个人在国家队的首球。2014年以12个联赛进球，蝉联中超本土最佳射手称号。2015年11月7日，武磊获颁本土最佳射手称号。2017年11月11日，当选2017中超联赛最佳本土射手。2016年，武磊获得亚洲足球先生提名。2017年11月，武磊连续第二年入围亚洲足球先生三甲候选之列。2018年3月8日，武磊在中超联赛对阵广州富力的比赛中上演“大四喜”。2018年3月，《世界足球》评选世界500强球员，武磊成唯一上榜中国人。2018年11月21日，武磊包揽2018赛季中超金球奖和金靴奖，并入选中超最佳阵容。2019年1月28日，加盟西甲皇家西班牙人足球俱乐部。"""
    print("context: ", c)
    print("question: ", "生日")
    print("predict: ", search_tagging(c, "生日"))
    print("*"*30)
    
    print("question: ", "职业")
    print("predict: ", search_tagging(c, "职业"))
    print("*"*30)
    
    print("question: ", "国籍")
    print("predict: ", search_tagging(c, "国籍"))
    print("*"*30)
    
    print("question: ", "出生地")
    print("predict: ", search_tagging(c, "出生地"))
    print("*"*30)
    
    print("question: ", "杂志书")
    print("predict: ", search_tagging(c, "杂志"))
    print("*"*30)
    
    print("question: ", "进球数")
    print("predict: ", search_tagging(c, "进球数"))
    print("*"*30)
    
