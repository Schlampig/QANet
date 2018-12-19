# -*- coding: utf-8 -*-

import random
import numpy as np
from tqdm import tqdm
from codecs import open
import json as json
from collections import Counter
import tensorflow as tf
from embed import *  # 调用embed文件内的embedding方法


# Get examples and eval_examples
###################################################################################################################
def convert_idx(text, tokens):
    # 统计text中每个token的span
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, word_counter, char_counter):
    # input: filename is the loading path
    #        datatype: train, dev, test
    # output: examples: 用于训练/测试的数据
    #         eval_examples: 用于评估训练/测试当前checkpoint效果的数据
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as f:
        source = json.load(f)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                # 读取context数据
                context = para["context"]
                context_tokens = para["segmented_context"]  # context_tokens = [token, token, ...]
                context_chars = []  # context_chars = [[char, char, ...], [char, char, ...], ...]
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])  # 统计context中当前token的词频，默认context中每个token都会在所有question中涉及
                    for char in token:
                        char_counter[char] += len(para["qas"])
                    context_chars.append(list(token))
                spans = convert_idx(context, context_tokens)  # spans = [(token_start, token_end), (t_start, t_end), ..]
       
                # 读取其他数据
                for qa in para["qas"]:
                    total += 1
                    question = qa["question"]
                    ques_tokens = qa["segmented_question"]  # ques_tokens = [token, token, ...]
                    ques_chars = []  # ques_chars = [[char, char, ...], [char, char, ...], ...]
                    for token in ques_tokens:  
                        word_counter[token] += 1  # 统计question中当前token的词频，默认question中每个token在当前question中出现了
                        for char in token:
                            char_counter[char] += 1
                        ques_chars.append(list(token))
    
                    # 读取label数据
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_texts.append(answer["text"])
                        answer_span = answer["answer_span"]
                        y1s.append(answer_span[0])  # 存放起止点坐标
                        y2s.append(answer_span[-1])
                    # 赋值给输出
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        random.shuffle(examples)
    return examples, eval_examples


# Get tfrecords and meta
###################################################################################################################
def text2index(example_now, w2i_dict, limit):
    # 把token转为w2i_dict中对应的index
    # input: example_now: 包含分词或分字的字典
    #        w2i_dict: 词典/字典
    #        limit: 词/字数上线
    # output: idx: 词/字的index阵
    if len(limit) == 2:  # 生成字index，即[[char, char, ...], [char, char, ...], ...], char为index
        idx = np.zeros([limit[0], limit[1]], dtype=np.int32)  # 初始化
        for i, token in enumerate(example_now):
            for j, char in enumerate(token):
                if j < limit[1]:
                    try:
                        idx[i, j] = w2i_dict[char]
                    except:
                        idx[i, j] = w2i_dict["--OOV--"]
    else:  # 生成词index，即[token, token, ...], token为index
        idx = np.zeros(limit[0], dtype=np.int32)
        for i, token in enumerate(example_now):
            try:
                idx[i] = w2i_dict[token]
            except:
                idx[i] = w2i_dict["--OOV--"]
    return idx


def build_features(config, examples, out_file, word2idx_dict, char2idx_dict, is_test=False):
    # 根据example中的词向量，最终生成用于模型训练的数据并存储
    # input: word2idx_dict，词典
    #        char2idx_dict, 字典（当config.use_char_emb=False时是另一个词典)
    # output：meta，统计篇章数量
    # 设定长度限制
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = config.test_ans_limit if is_test else config.ans_limit
    char_limit = config.char_limit

    # 过滤超过长度的值
    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    # 生成并存储data与label
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    n = 0
    for example in tqdm(examples):
        total += 1
        if filter_func(example):
            continue
            
        # 生成上下文与问题的词index矩阵
        context_idxs = text2index(example["context_tokens"], word2idx_dict, limit=[para_limit])
        ques_idxs = text2index(example["ques_tokens"], word2idx_dict, limit=[ques_limit])
        
        # 生成答案的起止坐标
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)
        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0
        
        # 生成上下文与问题的字index矩阵
        if config.use_char_emb:  # 提取字信息
            context_char_idxs = text2index(example["context_chars"], char2idx_dict, limit=[para_limit, char_limit])
            ques_char_idxs = text2index(example["ques_chars"], char2idx_dict, limit=[ques_limit, char_limit])
            feature={"context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                     "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                     "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                     "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                     "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                     "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                     "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))}
        else:  # 只提取词信息
            feature={"context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                     "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                     "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                     "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                     "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))}
        
        # 存如tfrecords里
        record = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(record.SerializeToString())

    meta["total"] = total
    writer.close()
    return meta


# Get tfrecords and meta
###################################################################################################################
def save_json(filename, obj):
    with open(filename, "w") as f:
        json.dump(obj, f)
    return None


def prepare(config):
    # word_counter与char_counter分别统计词频与字频
    word_counter, char_counter = Counter(), Counter()

    # 预处理并存储文件
    # 生成中间data
    print('Get train_examples, train_eval, dev_examples, dev_eval, test_examples, test_eval...')
    train_examples, train_eval = process_file(config.train_file, word_counter, char_counter)
    dev_examples, dev_eval = process_file(config.dev_file, word_counter, char_counter)
    test_examples, test_eval = process_file(config.test_file, word_counter, char_counter)
    # 生成词典与字典，此处要自定义使用哪个embedding策略，这些策略都存放在embed.py中
    print('Get word_emb_mat, word2idx_dict, char_emb_mat, char2idx_dict...')
    word_emb_mat, word2idx_dict = simple_embedding(word_counter, emb_file=config.pretrain_word_emb_file, limit=-1, vec_size=config.vec_size)
    char_emb_mat, char2idx_dict = simple_embedding(char_counter, emb_file=None, limit=-1, vec_size=config.char_dim)
    # 生成最终格式data并存储
    print('Create and record train, dev, test data...')
    _ = build_features(config, train_examples, config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # 存储数据
    save_json(config.train_eval_file, train_eval)
    save_json(config.dev_eval_file, dev_eval)
    save_json(config.test_eval_file, test_eval)
    save_json(config.dev_meta, dev_meta)
    save_json(config.test_meta, test_meta)
    save_json(config.word_emb_file, word_emb_mat)
    save_json(config.char_emb_file, char_emb_mat)
    save_json(config.word_dictionary, word2idx_dict)
    save_json(config.char_dictionary, char2idx_dict)
    return None
