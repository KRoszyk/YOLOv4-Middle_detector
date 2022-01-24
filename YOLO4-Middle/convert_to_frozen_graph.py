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
from tensorflow.compat.v1.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS

    if not YOLO_CUSTOM_WEIGHTS:
        print("Loading Darknet_weights from:", Darknet_weights)
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES, channels=6, modified=False)
        load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
    else:
        checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
        print("Loading custom weights from:", checkpoint)
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, channels=6, modified=True)
        yolo.load_weights(checkpoint)  # use custom weights

    return yolo


def convert_to_pb(yolo):
    yolo.save(f'./checkpoints/{YOLO_TYPE}-Tiny-{YOLO_INPUT_SIZE}')
    print(f"model saves to /checkpoints/{YOLO_TYPE}-Tiny-{YOLO_INPUT_SIZE}")


if __name__ == '__main__':
    model = Load_Yolo_model()
    convert_to_pb(model)
