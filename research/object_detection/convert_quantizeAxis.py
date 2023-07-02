import tensorflow as tf

test_images_dir = input("Enter path of test images in jpg format\n") or "/object_detection/out/Iran_plate_detection/test/"

saved_model_dir = input("Enter path of checkpoint save_model") or "/object_detection/models/research/object_detection/checkpoints/tflite/saved_model/"

model_prefix = input("Enter path and prefix name for saving the **_quantf.tflite model") or "mobilenet_v2_1.0_coco"

IMAGE_SIZE = input("Input Image size") or 320

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files( + '*.jpg')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [320, 320])
    image = tf.cast(image / 127.5 - 1, tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
tflite_model_quant = converter.convert()

with open(model_prefix+'_quantf.tflite', 'wb') as f:
  f.write(tflite_model_quant)
