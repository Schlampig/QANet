# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from prepare import prepare
from main import train, test


# Configuration
#################################################################################################
# Set paths
# --------------------------------------------------------------------------------------- #
train_file = "/users/seria/zhuyujin/datasets/general_qa_data/gen_train.json"
dev_file = "/users/seria/zhuyujin/datasets/general_qa_data/gen_dev.json"
test_file = "/users/seria/zhuyujin/datasets/general_qa_data/gen_test.json"

# 输出文件存放路径
target_dir = "data"  # 存放生成的各种预处理文件的地址
train_dir = "train"  # 存放训练生成的模型数据的地址
model_name = "QANets_basic"
dir_name = os.path.join(train_dir, model_name)
# 日志路径
log_dir = os.path.join(dir_name, "event")
# 模型路径
save_dir = os.path.join(dir_name, "model")
# 答案路径
answer_dir = os.path.join(dir_name, "answer")
answer_file = os.path.join(answer_dir, "answer.json")
# 输入数据路径
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
# 输出结果路径
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
# 词典路径
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
pretrain_word_emb_file = "/users/seria/zhuyujin/datasets/Tencent_AILab_ChineseEmbedding.txt"  # 自定义embedding的路径
pretrain_char_emb_file = "/users/seria/zhuyujin/datasets/Tencent_AILab_ChineseEmbedding.txt"

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(), dir_name)):
    os.mkdir(os.path.join(os.getcwd(), dir_name))
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

# Set flags
# --------------------------------------------------------------------------------------- #
flags = tf.flags
# settings for files
flags.DEFINE_string("mode", "train", "Running mode train/debug/test")  # 运行模式
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

flags.DEFINE_string("train_file", train_file, "Train source file")  # 原始数据集地址
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")  # 运行于tensorflow的数据集
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")  # hold-out数据集，QANet里train,dev与test分开评测
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")  # 记录dev与test数据集中样本数量
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")  # 答案数据集

# settings for embeddings
flags.DEFINE_boolean("use_char_emb", False, "Tokenize character or not")  # 是否分字，为False则不分
flags.DEFINE_string("pretrain_word_emb_file", pretrain_word_emb_file, "Input file for word embedding")  # 预训练好的embedding字典
flags.DEFINE_string("pretrain_char_emb_file", pretrain_char_emb_file, "Input file for char embedding")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")  # （生成的）用于QANet的embedding矩阵
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")  # （生成的）用于QANet的embedding字典
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")

# settings for word-vectors
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")  # 字向量维度
flags.DEFINE_integer("vec_size", 200, "Embedding dimension for char")  # 词向量维度
flags.DEFINE_integer("char_limit", 10, "Limit length for character")  # 词长
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")  # 用于滤掉频次低于设置的词/字
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")
flags.DEFINE_boolean("without_limit", True, "Release the limitation of word_limit.")  # 为True能处理变长句子

# setting for lengths
flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 100, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("test_ans_limit", 50, "Limit length for question in test file")

# settings for model
flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 1, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 80000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.2, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")


# Set GPU
# --------------------------------------------------------------------------------------- #
def set_gpu(gpu_ratio, gpu_id):
    assert isinstance(gpu_ratio, float) and isinstance(gpu_id, str)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = gpu_ratio 
    session = tf.Session(config=cfg)
    return None


# Run
#################################################################################################
def main(_):
    config = flags.FLAGS
    set_gpu(gpu_ratio=0.3, gpu_id='1')
    if config.mode == "train":
        train(config)
    elif config.mode == "prepare":
        prepare(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
