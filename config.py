# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from prepro import prepro
from main import train, test


# Configuration
#############################################################################################
# Set GPU
# ----------------------------------------------------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"

# Set path
# ----------------------------------------------------------------------------------------- #
home = os.path.expanduser(".")
train_file = "./toy_data/toy_train.json"
dev_file = "./toy_data/toy_dev.json"
test_file = "./toy_data/toy_test.json"
glove_word_file = "./toy_data/Tencent_AILab_ChineseEmbedding.txt"
target_dir = "data"  # 存放prepro生成的文件
train_dir = "train"  # 存放训练生成的数据，包括history, records等
model_name = "FRC"  # 模型名称

dir_name = os.path.join(train_dir, model_name)
log_dir = os.path.join(dir_name, "event")
save_dir = os.path.join(dir_name, "model")
answer_dir = os.path.join(dir_name, "answer")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
answer_file = os.path.join(answer_dir, "answer.json")
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
# ----------------------------------------------------------------------------------------- #
flags = tf.flags
# settings for directories
flags.DEFINE_string("mode", "train", "Running mode prepro/train/test")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

# settings for files
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")

# settings for embeddings
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")
flags.DEFINE_integer("glove_word_size", 8824330, "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 200, "Embedding dimension for Glove")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")

# settings for lengths
flags.DEFINE_integer("para_limit", 600, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 20, "Limit length for question")
flags.DEFINE_integer("ans_limit", 16, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 600, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 20, "Limit length for question in test file")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")

# settings for process
flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

# settings for model
flags.DEFINE_integer("batch_size", 16, "Batch size")
flags.DEFINE_integer("num_steps", 120000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.2, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 20, "Checkpoints for early stop")

# extensions (Uncomment corresponding code in download.sh to download the required data)
fasttext_file = os.path.join(home, "data", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", False, "Whether to use fasttext")


# Run
#############################################################################################
# Set path
# ----------------------------------------------------------------------------------------- #
def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "test":
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
