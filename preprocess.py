# -*- coding: utf-8 -*-

import numpy as np
import json as json
from tqdm import tqdm
import jieba


def word_tokenize(sent):
    doc = jieba.lcut(sent)
    return [token for token in doc]


def _get_word(word, word2idx_dict):
    # return 1即返回字典第1位的向量，那是UNK
    if word in word2idx_dict:
        return word2idx_dict[word]
    return 1

    
def _get_char(char, char2idx_dict):
    if char in char2idx_dict:
        return char2idx_dict[char]
    return 1
    
    
def preprocess(query, config, word2idx_dict, char2idx_dict, query_type='question'):
    # predict时，预处理query, query是question或context, 两者limit的限制不同
    assert isinstance(query, str)
    example = {}
    example['ques_tokens'] = word_tokenize(query)
    ans_limit = config.ans_limit
    
    if query_type is 'question':
        ques_limit = config.test_ques_limit
    elif query_type is 'context':
        ques_limit = config.test_para_limit
    else:
        ques_limit = -1
        
    if len(example["ques_tokens"]) > ques_limit:
        return None, None
#         raise ValueError("The inputed query's lengths are over the limit")
        
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token, word2idx_dict)
    
    if config.use_char_emb:
        example['ques_chars'] = [list(token) for token in example['ques_tokens']]
        char_limit = config.char_limit
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char, char2idx_dict)
    else:
        ques_char_idxs = None
        
    return ques_idxs, ques_char_idxs
    
