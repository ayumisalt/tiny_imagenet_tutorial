import pandas as pd
import os
import shutil

df = pd.read_csv("../tiny-imagenet-200/val/val_annotations.txt", sep="\t", header=None)

for index, data in df.iterrows():
    # print("../tiny-imagenet-200/val/" + data[1])
    os.makedirs("../tiny-imagenet-200/val/" + data[1] + "/images", exist_ok=True)
    # print("../tiny-imagenet-200/val/" + data[1] + "/images/" + data[0])
    shutil.copy(
        "../tiny-imagenet-200/val/" + "images/" + data[0],
        "../tiny-imagenet-200/val/" + data[1] + "/images/" + data[0],
    )
