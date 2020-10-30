"""
This files takes in uncut videos and returns an array of clips and exports clips to temp_clips.
"""
import os
import cv2
import subprocess
import numpy as np
from PIL import Image
import time
import pandas as pd


def get_dominant_colors(create_csv, create_image, n_colors, full_clip_path):
    cap = cv2.VideoCapture(full_clip_path)
    ret, img = cap.read()
    cap.release()

    small_image = Image.fromarray(img).resize((64, 64))
    pixels = np.float32(np.asarray(small_image).reshape(-1, 3))

    rgb_cube = dict()
    dominant_colors_rgb = []

    if create_csv:
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
                r_distances[r_point] = max((1-(abs(r-r_point)/72) ** 3), 0)
            for g_point in g_points:
                g_distances[g_point] = max((1-(abs(g-g_point)/72) ** 3), 0)
            for b_point in b_points:
                b_distances[b_point] = max((1-(abs(b-b_point)/72) ** 3), 0)

            for r_point in r_points:
                for g_point in g_points:
                    for b_point in b_points:
                        rgb_cube[f'({r_point}, {g_point}, {b_point})'] += r_distances[r_point] + g_distances[g_point] + b_distances[b_point]
    if create_image:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        for i in range(0, n_colors):
            color = palette[i]
            dominant_colors_rgb.append(((int(color[0]), int(color[1]), int(color[2])), int(counts[i])))

        dominant_colors_rgb.sort(key=lambda x: x[1], reverse=True)

    return rgb_cube, dominant_colors_rgb


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def main(create_csv=True, create_image=False, n_colors=5):
    start_time = time.time()
    video_folders = "temp_clips"
    video_dirs = [f for f in os.listdir(video_folders) if os.path.isdir(os.path.join(video_folders, f))]
    print(f'Found video paths: {video_dirs}')
    output_path = 'clip_data'

    # video_dirs = video_dirs[1:]
    # video_dirs = ["F:\CLOUD FILM MEDIA\PROXIES"]

    video_count = len(video_dirs)
    i = 0
    for video_dir in video_dirs:
        print(f'Analyzing: {video_dir}')
        i += 1
        clip_paths = [f for f in os.listdir(os.path.join(video_folders, video_dir)) if
                      os.path.isfile(os.path.join(video_folders, video_dir, f))]
        clip_count = len(clip_paths)
        j = 0
        video_color_data = []
        df = pd.DataFrame()
        for clip_path in clip_paths:
            j += 1
            if j % 10 == 0:
                print(f'Analyzing video {i} of {video_count}, clip {j} of {clip_count}')

            full_clip_path = os.path.join(video_folders, video_dir, clip_path)
            clip_data = dict()
            clip_data["fullpath"] = full_clip_path
            clip_data["length"] = get_length(filename=full_clip_path)
            rgb_cube, scene_ordered_colors_rgb = get_dominant_colors(create_csv, create_image, n_colors, full_clip_path)
            clip_data.update(rgb_cube)
            video_color_data.append(scene_ordered_colors_rgb)
            df = df.append(clip_data, ignore_index=True)

        if create_csv:
            df.to_csv(f'clip_data\{video_dir.split(".")[0]}.csv', index=False)

        if create_image:
            img_width = 4096
            img_height = 32 * len(clip_paths)
            video_color_list = []
            for clip_color_data in video_color_data:
                clip_color_list = []
                for color_data in clip_color_data:
                    for x in range(0, color_data[1]):
                        clip_color_list.append(color_data[0])
                for y in range(0, 32):
                    video_color_list += clip_color_list
            img = Image.new('RGB', (img_width, img_height))
            img.putdata(video_color_list)
            image_path = f'clip_images/{video_dir}-{n_colors}.png'
            # image_path = 'clip_images/SD.png'
            img.save(image_path)

        print(f'Time since start {time.time() - start_time}')


if __name__ == "__main__":
    main()
