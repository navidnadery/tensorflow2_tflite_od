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

input_size = 320
def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  img = tf.io.decode_image(img_data, channels=3)
  img = tf.image.resize(img, [input_size, input_size])
  #img = img / 127.5 - 1
  input_tensor = tf.expand_dims(img, 0)
  input_array = input_tensor.numpy()
  return input_array.astype(np.uint8)


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


pipeline_config = "/object_detection/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config"

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

interpreter = tf.lite.Interpreter(model_path="/home/tensorflow/models/research/mobilenet_v2_1.0_coco_quantf.tflite")
#interpreter = tf.lite.Interpreter(model_path="/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/quantized_ssdtfmodel.tflite")
#interpreter = tf.lite.Interpreter(model_path="/object_detection/models/research/object_detection/checkpoint/plt//tflite/model.tflite")
        #"/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tflite/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scale_s, zero_point_s = output_details[0]['quantization']
print(f"score scale {scale_s} zero {scale_s}")
scale_b, zero_point_b = output_details[1]['quantization']
print(f"boxes scale {scale_b} zero {zero_point_b}")
scale_c, zero_point_c = output_details[3]['quantization']
print(f"classes scale {scale_c} zero {zero_point_c}")
t2 = 0
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path("/object_detection/")
PATH_TO_TEST_IMAGES_DIR = pathlib.Path("/object_detection/out/vehicles/test")
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))[:20]
for i, test_image_path in enumerate(TEST_IMAGE_PATHS):
    t1 = time()
    image_path = str(test_image_path)
    image_np = load_image_into_numpy_array(image_path)
    #input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    #preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    #((interpreter.get_tensor(output_details[1]['index']) - zero_point_b) * scale_b * input_size).round(decimals=0, out=None)
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    t2 += time() - t1
    #print("scores")
    #print(scores)
    #print("classes")
    #print(classes)
    #print("boxes")
    #print(boxes)
    plot_detections(image_np[0], boxes, classes.astype(np.uint32) + label_id_offset, scores, category_index, figsize=(15, 20), image_name="/object_detection/gif_frame_" + ('%02d' % i) + ".jpg", threshold=0.2)

print(f"It took {t2} seconds")
