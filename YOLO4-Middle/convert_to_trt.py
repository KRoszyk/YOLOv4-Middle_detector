import os
import cv2
import numpy as np
import tensorflow as tf
from utils import *
from configs import *
from dataset import Dataset
from evaluate_mAP import get_mAP
from yolov4_f import Create_Yolo
from pathlib import Path
from natsort import natsorted
from tensorflow.python.compiler.tensorrt import trt_convert as trt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def calibration_input():
    for i in range(100):
        batched_input = np.random.random((1, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 6)).astype(np.float32)
        batched_input = tf.constant(batched_input)
        yield batched_input,


conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=4000000000)
conversion_params = conversion_params._replace(precision_mode=YOLO_TRT_QUANTIZE_MODE)
conversion_params = conversion_params._replace(max_batch_size=8)
if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    conversion_params = conversion_params._replace(use_calibration=True)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=f'./checkpoints/{YOLO_TYPE}-Tiny-{YOLO_INPUT_SIZE}',
                                    conversion_params=conversion_params)
if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    converter.convert(calibration_input_fn=calibration_input)
else:
    converter.convert()

converter.save(output_saved_model_dir=f'./checkpoints/{YOLO_TYPE}-Tiny-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}')
print(
    f'Done Converting to TensorRT, model saved to: /checkpoints/{YOLO_TYPE}-Tiny-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}')
