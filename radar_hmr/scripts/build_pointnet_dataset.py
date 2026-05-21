from radar_hmr.datasets.pointnet_builder import (
    PointNetDatasetBuilder
)

DATASET_ROOT = (
    r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1"
)

OUTPUT_DIR = (
    r"D:\Study_Resources\DRDO\Gesture_Recognition\processed\pointnet"
)

builder = PointNetDatasetBuilder(
    dataset_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
)

builder.build()