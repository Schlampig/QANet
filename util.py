# -*- coding: utf-8 -*-

import re
import string
import numpy as np
from tqdm import tqdm
from collections import Counter
import tensorflow as tf


def get_record_parser(config, is_test=False):
    # 解读config配置信息
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={"context_idxs": tf.FixedLenFeature([], tf.string),
                                                     "ques_idxs": tf.FixedLenFeature([], tf.string),
                                                     "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                                     "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                                     "y1": tf.FixedLenFeature([], tf.string),
                                                     "y2": tf.FixedLenFeature([], tf.string),
                                                     "id": tf.FixedLenFeature([], tf.int64)})
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
    return parse


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file)
    dataset = dataset.map(parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file)
    dataset = dataset.map(parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs):
            c_len = tf.reduce_sum(tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(key_func, reduce_func, window_size=5*config.batch_size))
        dataset = dataset.shuffle(len(buckets) * 25)

    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    # 把模型的输出结果（问题对应的文章id、答案起点、答案终点）转为答案字符串存入字典中输出（便于统计正确率）
    # answer_dict = {qa_id: answer(str), qa_id: answer(str), ...}
    # remapped_dict = {uuid: answer(str), uuid: answer(str), ...}
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def normalize_answer(s):
    # 清洗答案文本（英文版）
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():  # key即qa_id, value即预测的answer(str)
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run([model.qa_id, model.loss, model.yp1, model.yp2],
                                          feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]
