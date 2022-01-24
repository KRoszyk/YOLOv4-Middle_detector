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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass
    Darknet_weights = YOLO_V4_WEIGHTS
    if not YOLO_CUSTOM_WEIGHTS:
        print("Loading Darknet_weights from:", Darknet_weights)
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES, channels=6, modified=True)
        load_yolo_weights(yolo, Darknet_weights)  # use Darknet weights
    else:
        checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
        print("Loading custom weights from:", checkpoint)
        yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, channels=6, modified=True)
        yolo.load_weights(checkpoint)  # use custom weights

    return yolo


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_tensor = np.load(image_path)
    original_image = original_tensor[:, :, 0:3]
    original_image = np.array(original_image)

    image_data = image_preprocess(np.copy(original_tensor), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # pred_bbox = yolo.predict(image_data)
    batched_input = tf.constant(image_data)
    result = Yolo(batched_input)
    pred_bbox = []
    for key, value in result.items():
        value = value.numpy()
        pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_tensor, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

    if output_path != '':
        cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()

    return image


if __name__ == '__main__':
    images_folder = Path("data/tensors_6_channels")
    target_folder = "/data/predictions/yolo4_middle_predictions/"

    yolo = Load_Yolo_model()
    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, channels=6, modified=True)
    images_list = []

    for subfolder in images_folder.iterdir():
        for set_folder in subfolder.iterdir():
            for image in set_folder.iterdir():
                images_list.append(str(image))

    images_list.sort()

    for image in images_list:
        output_file = target_folder + str(image[-21:])
        output_file = output_file[:-4] + ".jpg"

        output_folder = output_file[:-11]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        detect_image(yolo, str(image), output_file, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES,
                     rectangle_colors=(255, 0, 0))

    print("End of creating images, now calculating MAP.")
    testset = Dataset('test')
    model_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, LOAD_MODEL_NAME)

    # Calculate value of mAP metric
    mAP_model.load_weights(model_directory)

