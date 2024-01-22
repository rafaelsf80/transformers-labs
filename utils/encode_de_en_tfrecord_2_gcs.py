""" Creates DE_EN input tfrecord file for batch inference and 
    overwrites the test split of wmt_t2t_ende in GCS 
    Input: INPUT_RAW_DATASET_FILE (data/dataset_raw_de_en.txt)
    Output: OUTPUT_GCS_FILE
"""

import os
import tensorflow as tf

INPUT_RAW_DATASET_FILE = "dataset_raw_de_en.txt"
OUTPUT_GCS_FILE = "gs://argolis-t5x-vertex-new/datasets/wmt_t2t_translate/de-en/1.0.0/wmt_t2t_translate-test.tfrecord-00000-of-00001"

with open(os.path.join("data", INPUT_RAW_DATASET_FILE), "r", encoding="utf-8") as f:
    all_text = f.readlines()
    all_text = [text.strip("\n").strip() for text in all_text]
    german_text = all_text[0::3] 
    english_text = all_text[1::3]  


# Create dataset in tfrecord and uploads to GCS
german_text_train = german_text[:int(len(german_text))]   
english_text_train = english_text[:int(len(english_text))]   
with tf.io.TFRecordWriter(OUTPUT_GCS_FILE) as writer:
    for german, english in zip(german_text_train, english_text_train):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'de': tf.train.Feature(bytes_list=tf.train.BytesList(value=[german.encode('utf-8')])),
            'en': tf.train.Feature(bytes_list=tf.train.BytesList(value=[english.encode('utf-8')]))
        }))   
        writer.write(tf_example.SerializeToString())
writer.close()