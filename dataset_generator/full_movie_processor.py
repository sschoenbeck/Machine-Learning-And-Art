import pandas as pd
import os
import cv2
import subprocess
import numpy as np
from PIL import Image
import time
import pickle
import json
from scenedetect import VideoManager, SceneManager, video_splitter
from scenedetect.detectors import ContentDetector


def find_scenes(video_path, threshold=40.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


def save_frame(full_clip_path, img_path, frame_number):
    cap = cv2.VideoCapture(full_clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, img = cap.read()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(img_path)
    cap.release()


def get_dominant_colors(img_path):
    img = Image.open(img_path)
    small_image = img.resize((64, 64))
    pixels = np.float32(np.asarray(small_image).reshape(-1, 3))

    rgb_cube = dict()
    r_points = [0, 128, 255]
    g_points = [0, 128, 255]
    b_points = [0, 128, 255]

    for r_point in r_points:
        for g_point in g_points:
            for b_point in b_points:
                rgb_cube[f'({r_point}, {g_point}, {b_point})'] = 0.0

    for pixel in list(pixels):
        r, g, b = pixel
        r_distances = dict()
        g_distances = dict()
        b_distances = dict()

        for r_point in r_points:
            r_distances[r_point] = max((1 - (abs(r - r_point) / 72) ** 3), 0)
        for g_point in g_points:
            g_distances[g_point] = max((1 - (abs(g - g_point) / 72) ** 3), 0)
        for b_point in b_points:
            b_distances[b_point] = max((1 - (abs(b - b_point) / 72) ** 3), 0)

        for r_point in r_points:
            for g_point in g_points:
                for b_point in b_points:
                    rgb_cube[f'({r_point}, {g_point}, {b_point})'] += r_distances[r_point] +\
                                                                      g_distances[g_point] + b_distances[b_point]
    return rgb_cube


def detect_objects(img_text_path, result_path):
    object_id_max = 80
    print('Detecting Objects')
    print(f'Input:\t{img_text_path}')
    print(f'Output:\t{result_path}')
    _ = subprocess.run(['darknet.exe', 'detector', 'test', 'cfg/coco.data', 'cfg/yolov4.cfg', 'yolov4.weights',
                        '-ext_output', '-dont_show', '-out', result_path, '<', img_text_path],
                       cwd="C:/Users/Simon/vcpkg/installed/x64-windows/tools/darknet",
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    results_json = json.load(open(result_path))
    object_df = pd.DataFrame()

    for result in results_json:
        object_dict = {'filename': result['filename'], 'object_labels': []}

        for i in range(0, object_id_max):
            object_dict[f'object_id_{str(i).rjust(2, "0")}'] = 0

        for detected_object in result['objects']:
            object_dict[f'object_id_{str(detected_object["class_id"]).rjust(2, "0")}'] += 1
            if detected_object['name'] not in object_dict['object_labels']:
                object_dict['object_labels'].append(detected_object['name'])
                
        object_df = object_df.append(object_dict, ignore_index=True)

    return object_df


def main():
    pd.set_option('display.max_columns', None)
    print(f'ffmepeg is available: {video_splitter.is_ffmpeg_available()}')
    print(f'mkvmerge is available: {video_splitter.is_mkvmerge_available()}')

    working_dir = os.getcwd()

    print(f'Work directory: {working_dir}')
    full_video_folder = working_dir + "\\temp\\full_videos"

    video_paths = [f for f in os.listdir(full_video_folder) if os.path.isfile(os.path.join(full_video_folder, f))]
    print(f'Found video paths: {video_paths}')
    video_paths = video_paths[10:]
    video_count = len(video_paths)
    i = 0
    for video_path in video_paths:
        i += 1
        video_name = video_path.split('.')[0]
        print(f'Cutting video {i} of {video_count}: {video_name}')
        full_video_path = os.path.join(full_video_folder, video_path)
        found_scenes = find_scenes(full_video_path)
        try:
            os.mkdir(f'temp_imgs\\{video_name}')
        except FileExistsError:
            pass
        movie_df = pd.DataFrame()
        scene_number = 0
        found_scene_count = len(found_scenes)

        img_text_path = working_dir + f'\\temp_imgs\\{video_name}\\image_list.txt'
        result_path = working_dir + f'\\temp_imgs\\{video_name}\\results.json'

        f = open(img_text_path, "w")
        for scene in found_scenes:
            if scene_number % 10 == 0:
                print(f'Analyzing scene: {scene_number} of {found_scene_count}')
            scene_dict = dict()
            start_timecode = scene[0]
            end_timecode = scene[1]

            scene_dict['start_time'] = start_timecode.get_seconds()
            scene_dict['start_frame'] = int(start_timecode.get_frames())
            scene_dict['end_time'] = end_timecode.get_seconds()
            scene_dict['end_frame'] = int(end_timecode.get_frames())
            scene_dict['length'] = end_timecode - start_timecode
            scene_dict['middle_frame'] = int((start_timecode.get_frames() + end_timecode.get_frames())/2)
            scene_dict['img_path'] = f'temp_imgs\\{video_name}\\{str(scene_number).rjust(4, "0")}.png'

            save_frame(full_video_path, scene_dict['img_path'], scene_dict['middle_frame'])
            rgb_cube = get_dominant_colors(scene_dict['img_path'])
            scene_dict.update(rgb_cube)
            movie_df = movie_df.append(scene_dict, ignore_index=True)

            img_path = working_dir + '\\' + scene_dict['img_path']
            img_path = img_path.replace('\\', '\\\\') + '\n'
            f.write(img_path)

            scene_number += 1

        f.close()
        movie_df.to_csv(f'temp_imgs\\{video_name}\\movie_data.csv', index=False)
        object_df = detect_objects(img_text_path, result_path)
        complete_df = pd.concat([movie_df, object_df], axis=1)
        complete_df.to_csv(f'clip_data\\{video_name}_complete_data.csv', index=False)


if __name__ == "__main__":
    main()
