import matplotlib.pyplot as plt

from radar_hmr.pipelines.radar_pipeline import (
    RadarPipeline
)

BIN_FILE = r"D:\Study_Resources\DRDO\Gesture_Recognition\New_Gesture_Dataset\Subject_1\LS\adc_data_ls2.bin"

pipeline = RadarPipeline(
    cfar_threshold=4
)

results = pipeline.process(
    BIN_FILE
)

points = results["points"]

plt.figure(figsize=(8, 8))

plt.scatter(
    points[:, 0],
    points[:, 1],
    c=points[:, 2],
    cmap='jet',
    s=20
)

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.title("Radar Pipeline Point Cloud")

plt.colorbar(label="Power")

plt.axis('equal')

plt.grid(True)

plt.show()