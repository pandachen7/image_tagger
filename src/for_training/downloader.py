"""
建議用另一個venv, 不然pip pkg會大洗牌
要先灌mongodb (docker可)
記得設定
export FIFTYONE_DATABASE_URI="mongodb://localhost:27017"
"""

import os

# pip install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

# 前面需補齊, 讓id能正確對應
my_classes = [
    "leo_cat",
    "ferret",
    "civet",
    "cat",
    "bicycle",
    "bird",
    "bus",
    "car",
    "dog",
    "motorbike",
    "person",
    "potted plant",
    "truck",
]


# 設定匯出的目標路徑
dataset = "coco-2017"
export_dir = f"/home/asys/datasets/img/{dataset}"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

dataset = foz.load_zoo_dataset(
    dataset,
    split="validation",
    label_types=[
        "detections",  # default
        #  "segmentations"
    ],
    classes=[
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        # "boat",
        # "traffic light",
        # "fire hydrant",
        # "stop sign",
        # "parking meter",
        # "bench",
        "bird",
        "cat",
        "dog",
        # "horse",
        # "sheep",
        # "cow",
        # "elephant",
        # "bear",
        # "zebra",
        # "giraffe",
        # "backpack",
        # "umbrella",
        # "handbag",
        # "tie",
        # "suitcase",
        # "frisbee",
        # "skis",
        # "snowboard",
        # "sports ball",
        # "kite",
        # "baseball bat",
        # "baseball glove",
        # "skateboard",
        # "surfboard",
        # "tennis racket",
        # "bottle",
        # "wine glass",
        # "cup",
        # "fork",
        # "knife",
        # "spoon",
        # "bowl",
        # "banana",
        # "apple",
        # "sandwich",
        # "orange",
        # "broccoli",
        # "carrot",
        # "hot dog",
        # "pizza",
        # "donut",
        # "cake",
        # "chair",
        # "couch",
        "potted plant",
        # "bed",
        # "dining table",
        # "toilet",
        # "tv",
        # "laptop",
        # "mouse",
        # "remote",
        # "keyboard",
        # "cell phone",
        # "microwave",
        # "oven",
        # "toaster",
        # "sink",
        # "refrigerator",
        # "book",
        # "clock",
        # "vase",
        # "scissors",
        # "teddy bear",
        # "hair drier",
        # "toothbrush"
    ],
    max_samples=2000,
)

dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    classes=my_classes,
)
