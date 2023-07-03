import tensorflow as tf

# Load the model
interpreter = tf.lite.Interpreter(model_path='ssd_mobilenet_v2_coco_quant_postprocess.tflite')
#interpreter = tf.lite.Interpreter(model_path='/object_detection/models/research/object_detection/checkpoint/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/quantized_ssdtfmodel.tflite')

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print the input details
for i, input_detail in enumerate(input_details):
    print(f'Input {i}:')
    print(f'  name: {input_detail["name"]}')
    print(f'  shape: {input_detail["shape"]}')
    print(f'  dtype: {input_detail["dtype"]}')

# Print the output details
for i, output_detail in enumerate(output_details):
    print(f'Output {i}:')
    print(f'  name: {output_detail["name"]}')
    print(f'  shape: {output_detail["shape"]}')
    print(f'  dtype: {output_detail["dtype"]}')
