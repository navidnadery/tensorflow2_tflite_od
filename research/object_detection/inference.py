import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
import tensorflow as tf
from time import time
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list


model_dir = "/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_320x320_coco17_tpu-8"
pipeline_config = "/object_detection/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config"

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-31')).expect_partial()


def get_model_detection_function(model):
  """Get a tf.function for detection."""
  @tf.function
  def detect_fn(image):
    """Detect objects in image."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])
  return detect_fn
  

detect_fn = get_model_detection_function(detection_model)

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path("/object_detection/out/vehicles/test/")
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))[:20]

t2 = 0
label_id_offset = 1
for i, test_image_path in enumerate(TEST_IMAGE_PATHS):
    t1 = time()
    image_path = str(test_image_path)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    t2 += time() - t1
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
        keypoints = detections['detection_keypoints'][0].numpy()
        keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
    viz_utils.visualize_boxes_and_labels_on_image_array(image_np, detections['detection_boxes'][0].numpy(), (detections['detection_classes'][0].numpy() + label_id_offset).astype(int), detections['detection_scores'][0].numpy(), category_index, use_normalized_coordinates=True, max_boxes_to_draw=200, min_score_thresh=.5, agnostic_mode=False, keypoints=keypoints, keypoint_scores=keypoint_scores, keypoint_edges=get_keypoint_tuples(configs['eval_config']))
    plt.figure(figsize=(12,16))
    plt.imsave(f"/object_detection/orig_{str(i).zfill(2)}.jpg", image_np)

print(f"It took {t2} seconds")
