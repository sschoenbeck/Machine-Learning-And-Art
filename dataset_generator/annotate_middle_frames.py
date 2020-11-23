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
    output_img_folder = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/annotated_middle_frames'
    img_dirs = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    print(img_dirs)
    img_text_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames.txt'
    img_dirs = img_dirs[682:]
    i = 0
    img_count = len(img_dirs)
    for img_path in img_dirs:
        full_img_path = img_folder + '/' + img_path
        annotated_img_path = output_img_folder + '/' + img_path.split('.')[0] + '-annotated.jpg'

        subprocess.run(['darknet.exe', 'detector', 'test', 'cfg/coco.data', 'cfg/yolov4.cfg', 'yolov4.weights',
                        '-dont_show'],
                       cwd="C:/Users/Simon/vcpkg/installed/x64-windows/tools/darknet",
                       shell=True,
                       input=full_img_path.encode('utf-8'),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
        os.rename(r'C:/Users/Simon/vcpkg/installed/x64-windows/tools/darknet/predictions.jpg', annotated_img_path)
        i += 1
        print(f'{i} of {img_count}: {annotated_img_path}')
    # 'C:/Users/Simon/vcpkg/installed/x64-windows/tools/darknet'
    # result_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames.json'


if __name__ == "__main__":
    main()
