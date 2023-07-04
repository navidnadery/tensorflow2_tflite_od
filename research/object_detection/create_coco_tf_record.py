# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
import PIL.Image
import xmltodict
from glob import glob
from bs4 import BeautifulSoup

from pycocotools import mask
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.compat.v1.app.flags
tf.compat.v1.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')
tf.compat.v1.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.compat.v1.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.compat.v1.flags.DEFINE_string('train_annotations_dir', '',
                       'Training annotations directory.')
tf.compat.v1.flags.DEFINE_string('test_annotations_dir', '',
                       'Test annotations directory.')
tf.compat.v1.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  try:
      with tf.compat.v1.gfile.GFile(full_path, 'rb') as fid:
          encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = PIL.Image.open(encoded_jpg_io)
      key = hashlib.sha256(encoded_jpg).hexdigest()
  except:
      print(full_path)

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  encoded_mask_png = []
  num_annotations_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(0)
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id].encode('utf8'))

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)
      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())
  feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(0),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return key, example, num_annotations_skipped


def _create_tf_record_from_coco_annotations(
    annotations_dir, image_dir, output_path, include_masks, num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
  """
  idx = 0
  annotations_index = {}
  annotations = glob(annotations_dir+'/*.xml')
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)

    category_index = {1: 'lpr'} #{3: 'car', 6: 'bus', 8: 'truck'}
    category_index_ = {v: k for k, v in category_index.items()}

    total_num_annotations_skipped = 0
    for iid, annotation in enumerate(annotations):
        bs_data = xmltodict.parse(open(annotation).read())['annotation']
        width, height = 300 / int(bs_data['size']['width']), 300 / int(bs_data['size']['height'])
        try:
            objects = bs_data['object']
            if type(objects) is not list:
                objects = [objects]
            annotations_list = []
        except:
            continue
        for obj in objects:
            if idx % 100 == 0:
                tf.compat.v1.logging.info('On image %d of %d', idx, len(annotations))
            # try:
            if True:
                dx = int(obj['bndbox']['xmax'])-int(obj['bndbox']['xmin'])
                dy = int(obj['bndbox']['ymax'])-int(obj['bndbox']['ymin'])
                if dx * width < 3 or dy * height < 3:
                    continue
                annotations_list.append({'bbox': (max(0, int(obj['bndbox']['xmin'])), max(0, int(obj['bndbox']['ymin'])), dx, dy), 'category_id': category_index_[obj['name']]})
            # except:
            #     pass
            idx += 1
        image = {k: int(v) for k, v in bs_data['size'].items()}
        image['id'] = iid
        try:
            image['file_name'] = bs_data['filename']
        except:
            pass
        _, tf_example, num_annotations_skipped = create_tf_example(
                    image, annotations_list, image_dir, category_index, include_masks)
        shard_idx = idx % num_shards
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tf.compat.v1.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.test_image_dir, '`test_image_dir` missing.'
  assert FLAGS.train_annotations_dir, '`train_annotations_dir` missing.'
  assert FLAGS.test_annotations_dir, '`test_annotations_dir` missing.'

  if not tf.compat.v1.gfile.IsDirectory(FLAGS.output_dir):
    tf.compat.v1.gfile.MakeDirs(FLAGS.output_dir)
  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  testdev_output_path = os.path.join(FLAGS.output_dir, 'coco_testdev.record')

  _create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_dir,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      num_shards=100)
  _create_tf_record_from_coco_annotations(
      FLAGS.test_annotations_dir,
      FLAGS.test_image_dir,
      testdev_output_path,
      FLAGS.include_masks,
      num_shards=100)


if __name__ == '__main__':
  tf.compat.v1.app.run()
