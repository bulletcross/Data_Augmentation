# Authored by: bulletcross@gmail.com
import tensorflow as tf
import numpy as np

GPU_ENABLED = False
NUM_EPOCHS = 10
PREFETCH_BUFFER = 1024
BATCH_SIZE = 64

def tfrecord_parser(record_file_path):
    # Describe the schema used for example proto dataset
    parse_op = tf.parse_single_example(serialized = record_file_path,
                features = {
                    'raw_img': tf.FixedLenFeature([], tf.string),
                    'label'  : tf.FixedLenFeature([], tf.int64)
                },
                name = 'parse_op')
    # Revert from string byte to int32
    img = tf.decode_raw(parse_op['raw_img'], tf.uint8, name = 'byte_to_int8_op')
    img = tf.cast(img, tf.float32, name = 'int8_to_int32_op')
    img = tf.reshape(img, shape=[128, 128, 3], name = 'reshape_op')
    label = tf.cast(parse_op['label'], tf.int32, name = 'label_cast_op')
    return img, label

def train_data_pipeline(train_data_file):
    dataset = tf.data.TFRecordDataset(filenames = train_data_file,
                                        num_parallel_reads = 8)
    dataset = dataset.apply(
                tf.contib.data.shuffle_and_repeat(PREFETCH_BUFFER, NUM_EPOCHS))
    dataset = dataset.apply(
                tf.contrib.data.map_and_batch(tfrecord_parser, BATCH_SIZE))
    if GPU_ENABLED:
        dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    return dataset

def test_data_pipeline(test_data_file):
    dataset = tf.data.TFRecordDataset(filenames = test_data_file,
                                        num_parallel_reads = 8)
    dataset = dataset.apply(
                tf.contrib.data.map_and_batch(tfrecord_parser, BATCH_SIZE))
    if GPU_ENABLED:
        dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))
    return dataset

def data_pipeline(mode = 'train_test', train_file = None, test_file = None):
    if mode == 'train':
        if train_file == None:
            print("No filename has been passed")
            return None
        train_dataset = train_data_pipeline(train_file)
        #iterator = dataset.make_initializable_iterator()
        iterator = train_dataset.make_one_shot_iterator()
        img_batch, label_batch = iterator.get_next()
        return img_batch, label_batch
    elif mode == 'test':
        if test_file == None:
            print("No test file has been passed")
            return None
        test_dataset = test_data_pipeline(test_file)
        iterator = test_dataset.make_one_shot_iterator()
        img_batch, label_batch = iterator.get_next()
        return img_batch, label_batch
    else:
        if train_file == None or test_file == None:
            print("No file name has been passed")
            return None
        train_dataset = train_data_pipeline(train_file)
        test_dataset = test_data_pipeline(test_file)
        generic_iterator = tf.data.Iterator.from_structure(
                            train_dataset.output_types,
                            train_dataset.output_shapes)
        train_init_op = generic_iterator.make_initializer(train_dataset)
        test_init_op = generic_iterator.make_initializer(test_dataset)
        img_batch, labels_batch = generic_iterator.get_next()
        return img_batch, label_batch, train_init_op, test_init_op
