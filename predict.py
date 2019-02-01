# -*- coding: utf-8 -*-

import numpy as np
import json as json
import tensorflow as tf

from config import flags
from model import Model
from preprocess import preprocess


# Configuration
#############################################################################################
config = flags.FLAGS
graph = tf.Graph()
with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
with open(config.word_dictionary, "r") as fh:
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
    answer = ''.join(seg_c[yp1[0]: yp2[0]])
    return answer


if __name__ == '__main__':

    c = """就读初中二年级的鹿目圆，过着平凡幸福的生活。神秘转学生晓美焰的出现，开始让小圆的命运有了巨大转变。某日一只名为丘比的神秘生物，希望小圆能够与它签订魔法契约，成为“魔法少女”以对抗邪恶的魔女保护世界。正当小圆犹豫烦恼之时，好友沙耶香先一步成为“魔法少女”后，两人才发现原来签订契约后，需付出的代价远比远比想象中巨大甚至残酷，这一切真相都再次冲击小圆想成为“魔法少女”的想法。"""
    q = "谁是小圆的朋友？"

    print("context: ", c)
    print("question: ", q)
    print("prediction: ", search_tagging(c, q))
