""" Creates two tfrecord files (train and eval) to use in finetuning """

import os
import tensorflow as tf

RAW_DATASET_FILE = "dataset_raw_tr_en.txt"
TRAIN_EVAL_SPLIT = 0.95 # 95%

with open(os.path.join("data", RAW_DATASET_FILE), "r", encoding="utf-8") as f:
    all_text = f.readlines()
    all_text = [text.strip("\n").strip() for text in all_text]
    turkish_text = all_text[0::4] 
    english_text = all_text[2::4]  


# Train dataset in tfrecord
turkish_text_train = turkish_text[:int(len(turkish_text)*TRAIN_EVAL_SPLIT)]   
english_text_train = english_text[:int(len(english_text)*TRAIN_EVAL_SPLIT)]   
with tf.io.TFRecordWriter("data/train.tfrecords") as writer:
    for turkish, english in zip(turkish_text_train, english_text_train):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'turkish': tf.train.Feature(bytes_list=tf.train.BytesList(value=[turkish.encode('utf-8')])),
            'english': tf.train.Feature(bytes_list=tf.train.BytesList(value=[english.encode('utf-8')]))
        }))   
        writer.write(tf_example.SerializeToString())
writer.close()


# Eval dataset in tfrecord
turkish_text_test = turkish_text[int(len(turkish_text)*TRAIN_EVAL_SPLIT):]   
english_text_test = english_text[int(len(english_text)*TRAIN_EVAL_SPLIT):]   
with tf.io.TFRecordWriter("data/eval.tfrecords") as writer:
    for turkish, english in zip(turkish_text_test, english_text_test):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'turkish': tf.train.Feature(bytes_list=tf.train.BytesList(value=[turkish.encode('utf-8')])),
            'english': tf.train.Feature(bytes_list=tf.train.BytesList(value=[english.encode('utf-8')]))
        }))   
        writer.write(tf_example.SerializeToString())
writer.close()
 




