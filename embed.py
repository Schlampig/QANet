# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm


def simple_embedding(counter, emb_file=None, limit=-1, vec_size=25):
    # 直接调用预训练模型的embedding策略
    # input: counter = {token: count, token:count}, 统计词频
    #        emb_file, 预训练好的词向量json文件，内部为字典，token -> [token, vector]
    #        limit, 只保留count大于limit的token
    #        vec_size: 词向量维度
    # output: emb_mat is a 2D list, emb_mat = [vec, vec, ...], where vec is an vector, embedding矩阵
    #         token2idx_dict is a dict, token2idx_dict = {token: index, token: index, ...}, embedding字典
    assert isinstance(vec_size, int) and vec_size <= 200
    embedding_dict = {}  # embedding_dict = {word: vec, word: ver, ...}
    if emb_file is not None:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = array[0]  # 第一个为词本身
                vector = list(map(float, array[-vec_size:])) # 后面的词为词向量
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
    else:  # 随机生成vector
        filtered_elements = [k for k, v in counter.items() if v > limit]
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    # 生成token2idx_dict, idx2emb_dict, 与emb_mat
    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), start=2)}  # 前两个位置给NULL与OOV
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict
