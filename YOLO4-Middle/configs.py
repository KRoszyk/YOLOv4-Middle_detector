# Main options
YOLO_TYPE = "yolov4"
YOLO_FRAMEWORK = "tf"  # "tf" or "trt"
YOLO_V4_WEIGHTS = "/model_data/yolov4.weights"
YOLO_V4_TINY_WEIGHTS = "/model_data/yolov4-tiny.weights"
YOLO_CUSTOM_WEIGHTS = True  # if not using leave False, YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_STRIDES = [8, 16, 32]
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416
if YOLO_TYPE == "yolov4":
    YOLO_ANCHORS = [[[12, 16], [19, 36], [40, 28]],
                    [[36, 75], [76, 55], [72, 146]],
                    [[142, 110], [192, 243], [459, 401]]]
YOLO_COCO_CLASSES = "/model_data/coco/coco.names"

# Train options
TRAIN_YOLO_TINY = True
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = True  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES = "/labels/kaist_classes.names"
TRAIN_ANNOT_PATH = "/labels/yolo4_middle_train_labels.txt"
TRAIN_LOGDIR = "log"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = "yolo4_middle"
LOAD_MODEL_NAME = "yolov4_middle"
TRAIN_LOAD_IMAGES_TO_RAM = False  # With True faster training, but need more RAM
TRAIN_BATCH_SIZE = 32
TRAIN_INPUT_SIZE = 416
TRAIN_DATA_AUG = True
TRAIN_TRANSFER = True
TRAIN_FROM_CHECKPOINT = False
TRAIN_LR_INIT = 1e-5
TRAIN_LR_END = 1e-7
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS = 20
YOLO_TRT_QUANTIZE_MODE = "INT8"

# Test options
TEST_ANNOT_PATH = "/labels/yolo4_middle_test_labels.txt"
TEST_BATCH_SIZE = 32
TEST_INPUT_SIZE = 416
TEST_DATA_AUG = False
TEST_DECTECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45

if TRAIN_YOLO_TINY:
    YOLO_STRIDES = [16, 32]
    # YOLO_ANCHORS = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights
    YOLO_ANCHORS = [[[10, 14], [23, 27], [37, 58]],
                    [[81, 82], [135, 169], [344, 319]]]
