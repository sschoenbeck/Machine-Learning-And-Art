import os
import cv2
import subprocess
import numpy as np
from PIL import Image
import time
import pandas as pd
import pickle
import json


def main():
    img_folder = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames'
    img_dirs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    print(img_dirs)
    img_text_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames.txt'
    f = open(img_text_path, "w")

    for img_path in img_dirs:
        f.write(f'{img_folder}/{img_path}\n')
        print(img_path)

    f.close()
    result_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames.json'
    output = subprocess.run(['darknet.exe', 'detector', 'test', 'cfg/coco.data', 'cfg/yolov4.cfg', 'yolov4.weights',
                             '-ext_output', '-dont_show', '-out', result_path, '<', img_text_path],
                            cwd="C:/Users/Simon/vcpkg/installed/x64-windows/tools/darknet",
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    print(output)


if __name__ == "__main__":
    main()
