""" Sample code to decode tfrecord files """

import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset("wmt_t2t_translate-test.tfrecord-00000-of-00001")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)