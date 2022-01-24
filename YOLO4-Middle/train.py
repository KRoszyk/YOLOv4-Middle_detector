import os
import tensorflow as tf

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.python.client import device_lib
import shutil
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# from tensorflow.keras.utils import plot_model
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.98
config.gpu_options.allow_growth = True
from yolov4_f import Create_Yolo, compute_loss
from utils import load_yolo_weights
from configs import *
from evaluate_mAP import get_mAP
from dataset import Dataset

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if TRAIN_YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"

channels = 6


def return_number(name):
    name = name.replace("2d", "d")
    number = ''.join(filter(lambda i: i.isdigit(), name))
    if number == '':
        number = int(0)
    else:
        number = int(number)
    return number


def main():
    global TRAIN_FROM_CHECKPOINT

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)

    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    trainset = Dataset('train')
    testset = Dataset('test')
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # Making transfer learning from Darknet
    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES, channels=channels - 3,
                              modified=False)
        load_yolo_weights(Darknet, Darknet_weights)  # use darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES, channels=channels,
                       modified=True)

    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(f"./checkpoints/{LOAD_MODEL_NAME}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    not_pretrained_weights = []
    loaded_weights = []

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        print("Loading weights from Darknet!")
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            for idx, yolo_layer in enumerate(yolo.layers):
                yolo_name = yolo_layer.name
                darknet_name = l.name
                yolo_number = return_number(yolo_name)
                darknet_number = return_number(darknet_name)
                yolo_letters = ''.join([i for i in yolo_name if not i.isdigit()])
                yolo_letters = yolo_letters.replace("_", "")

                darknet_letters = ''.join([i for i in darknet_name if not i.isdigit()])
                darknet_letters = darknet_letters.replace("_", "")
                if layer_weights and (yolo_name == darknet_name or (
                        yolo_number == darknet_number + 78 and yolo_letters == darknet_letters) or
                                      (yolo_number == darknet_number + 81 and yolo_letters == darknet_letters)):
                    print(f'Darknet: {l.name}, yolo: {yolo_layer.name}')
                    try:
                        yolo_layer.set_weights(layer_weights)
                    except:
                        print("layers don't fit!")
                    else:
                        print("weights fit!")
                        loaded_weights.append(yolo_layer.name)

    for layer in yolo.layers:
        if layer.name not in loaded_weights:
            not_pretrained_weights.append(layer.name)

    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:  # and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, channels=channels, modified=True)  # create second model to measure mAP

    for layer in yolo.layers:
        if layer.name in not_pretrained_weights:
            layer.trainable = True
        else:
            layer.trainable = False

    yolo.compile()

    best_val_loss = 1000  # should be large at start

    for epoch in range(TRAIN_EPOCHS):
        if epoch == TRAIN_WARMUP_EPOCHS:
            for layer in yolo.layers:
                layer.trainable = True
            print("Unfreeze all layers!")

        if any(layer.trainable == False for layer in yolo.layers):
            print("Some layers are still frozen!")
        else:
            print("There are not frozen layers!")

        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0] % steps_per_epoch
            print(
                "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f},"
                " prob_loss:{:7.2f}, total_loss:{:7.2f}"
                    .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4],
                            results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue

        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
        validate_writer.flush()

        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER,
                                          TRAIN_MODEL_NAME + "_val_loss_" + str(total_val / count) + "_epoch_" + str(
                                              epoch))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss > total_val / count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val / count
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)

        # measure mAP of trained custom model
        try:
            mAP_model.load_weights(save_directory)  # use keras weights
            map_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
            map = get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
            with map_writer.as_default():
                tf.summary.scalar("map/total_map", float(map), step=epoch)
        except UnboundLocalError:
            print(
                "You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY "
                "and TRAIN_SAVE_CHECKPOINT lines in configs.py")


if __name__ == '__main__':
    main()
