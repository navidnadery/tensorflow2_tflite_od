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
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  image = image.resize((300,300))
  im_height, im_width = 300, 300
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None,
                    threshold=0.5):
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=threshold)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


pipeline_config = "/object_detection/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config"

num_classes = 3
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
#model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(model_config=model_config, is_training=True)

label_id_offset = 1
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)

interpreter = tf.lite.Interpreter(model_path="/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

t2 = 0
PATH_TO_TEST_IMAGES_DIR = pathlib.Path("/object_detection/out/vehicles/test/")
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))[:20]
for i, test_image_path in enumerate(TEST_IMAGE_PATHS):
    t1 = time()
    image_path = str(test_image_path)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])
    t2 += time() - t1
    plot_detections(image_np, boxes[0], classes[0].astype(np.uint32) + label_id_offset, scores[0], category_index, figsize=(15, 20), image_name="/object_detection/gif_frame_" + ('%02d' % i) + ".jpg", threshold=0.2)

print(f"It took {t2} seconds")
