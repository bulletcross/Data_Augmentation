# Authored by: bulletcross@gmail.com

import tensorflow as tf
import numpy as np
import cv2
import skimage.io as io
from skimage.transform import rescale, resize
import data_pipeline as dp

data_path = "Dataset/dataset_bin.tfrecords"

def tfrecord_parser(record_file_path):
    # Describe the schema used for example proto dataset
    parse_op = tf.parse_single_example(serialized = record_file_path,
                features = {
                    'raw_img': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                },
                name = 'parse_op')
    # Revert from string byte to int32
    img = tf.decode_raw(parse_op['raw_img'], tf.uint8, name = 'byte_to_int8_op')
    img = tf.reshape(img, shape=[128, 128, 3], name = 'reshape_op')
    label = tf.cast(parse_op['label'], tf.int32, name = 'label_cast_op')
    return img, label

def get_iterator():
    dataset = tf.data.TFRecordDataset(filenames = [data_path],
                                        num_parallel_reads = 2)
    dataset = dataset.map(tfrecord_parser)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    img, label = iterator.get_next()
    return img, label, iterator

def main():
    img, label, iter = get_iterator()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        while True:
            out_img, out_label = sess.run([img, label])
            print(str(out_label))
            print(out_img[0].shape)
            cv2.imshow('image', out_img[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__=='__main__':
    main()
