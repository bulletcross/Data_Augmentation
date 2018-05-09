# Authored by: bulletcross@gmail.com

import tensorflow as tf
import numpy as np
import glob
import cv2
import skimage.io as io
from skimage.transform import rescale, resize
from tqdm import tqdm

import random
import sys
import os

dataset_path = 'Dataset/train/*.jpg'
out_file = 'Dataset/dataset_bin.tfrecords'

"""
Note: This is not augmnetation on fly, this does create static
augmented data tfrecord file
For on the fly augmentation, train_data_pipeline in data_pipeline
module has to be worked upon.
"""

def process_image(img_path):
    img = cv2.imread(img_path)
    #img = io.imread(img_path)
    if img is None:
        print("Unable to read " + img_path)
        return None
    # Convert back to RGB
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Start preprocessing here, resizing with biliner interpolation
    #img = resize(img, (128, 128))
    # End of preprocessing, return img
    return img

def img_augment(img, options):
    # Options is a list of options for img augmentation
    img_list = [img] # TO-DO
    return img_list

def prepare_tfrecord(img_path_list, labels):
    print("Preparing tfrecord dataset....")
    writer = tf.python_io.TFRecordWriter(out_file)
    for i in tqdm(range(len(img_path_list))):
        img = process_image(img_path_list[i])
        label = labels[i]
        if img is None:
            continue
        aug_img_list = img_augment(img, options = {})
        for aug_img in aug_img_list:
            # Create Example proto buffer and write after serializing
            example = tf.train.Example(
                    features=tf.train.Features(
                        feature = {
                            'raw_img': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[aug_img.tostring()])
                            ),
                            'label': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label])
                            )
                        }
                    )
            )
            # Serialize example proto and write
            writer.write(example.SerializeToString())
    print("All image processed.")
    writer.close()
    sys.stdout.flush()

def img_label_list():
    file_list = glob.glob(dataset_path)
    labels = [0 if 'cat' in file_name or 'Cat' in file_name else 1 for file_name in file_list]
    return file_list, labels

def main():
    file_list, labels = img_label_list()
    prepare_tfrecord(file_list, labels)

if __name__=='__main__':
    main()
