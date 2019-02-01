# -*- coding: utf-8 -*-

import numpy as np
import jieba

jieba.add_word("XXX")


def word_tokenize(sent):
    return jieba.lcut(sent)


def _get_word(word, word2idx_dict):
    if word in word2idx_dict:
        return word2idx_dict[word]
    return 1


def preprocess(query, config, word2idx_dict, query_type=None):
    assert isinstance(query, str)
    seg_query = word_tokenize(query)
    # filter via limited length
    if query_type is "question":
        ques_limit = config.test_ques_limit
    elif query_type is "context":
        ques_limit = config.test_para_limit
    else:
        ques_limit = len(seg_query)

    if len(seg_query) > ques_limit:
        return None, None
    # generate indexes data
    ques_idxs = np.zeros([1, ques_limit], dtype=np.int32)
    for i, token in enumerate(seg_query):
        ques_idxs[0, i] = _get_word(token, word2idx_dict)
    return ques_idxs, seg_query
