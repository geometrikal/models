"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record



  NOTE see vertigo-processing for generating TF records from a ViaProject











"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import json
import numpy as np

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict



flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_dir', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def convert_str_to_dict(s: str):
    j = json.loads(s)
    return j


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'COTS':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):

    CLS_NAME = 'COTS'
    CLS = 1

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        boxes = convert_str_to_dict(row['region_shape_attributes'])
        if boxes:
            xmins.append(boxes['x'] / width)
            ymins.append(boxes['y'] / height)
            xmaxs.append((boxes['x'] + boxes['width']) / width)
            ymaxs.append((boxes['y'] + boxes['height']) / height)
            classes_text.append(CLS_NAME.encode('utf8'))
            classes.append(CLS)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    print(examples)
    grouped = split(examples, 'filename')
    tf_examples = []

    for group in grouped:
        tf_example = create_tf_example(group, path)
        tf_examples.append(tf_example)

    print(len(tf_examples))

    tf_examples = np.random.permutation(tf_examples)
    split_index = int(len(tf_examples) * 0.9)
    train_tf_examples = tf_examples[0:split_index]
    test_tf_examples = tf_examples[split_index:]
    train_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'train.record'))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_dir, 'test.record'))

    print(os.path.join(FLAGS.output_dir, 'train.record'))
    for record in train_tf_examples:
        train_writer.write(record.SerializeToString())
    for record in test_tf_examples:
        test_writer.write(record.SerializeToString())

    train_writer.close()
    test_writer.close()
    print('Successfully created the TFRecords')


if __name__ == '__main__':
    tf.app.run()
