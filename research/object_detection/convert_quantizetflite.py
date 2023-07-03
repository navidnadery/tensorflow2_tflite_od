import tensorflow as tf

test_images_dir = input("Enter path of test images in jpg format\n") or "/object_detection/out/Iran_plate_detection/test/"

saved_model_dir = input("Enter path of checkpoint save_model\n") or "/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/"

model_prefix = input("Enter path and prefix name for saving the **_quant.tflite model\n") or "mobilenet_v2_1.0_coco"

IMAGE_SIZE = input("Input Image size\n") or 320

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(test_images_dir + '*.jpg')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open(model_prefix+'_quant.tflite', 'wb') as f:
  f.write(tflite_model)
