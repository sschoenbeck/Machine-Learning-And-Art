"""
This files takes in uncut videos and returns an array of clips and exports clips to temp_clips.
"""
import io
import os
import cv2
import subprocess
import numpy as np
from PIL import Image
import time
import colorsys


def get_dominant_colors(full_clip_path, n_colors):
    cap = cv2.VideoCapture(full_clip_path)
    ret, img = cap.read()
    cap.release()

    small_image = Image.fromarray(img).resize((64, 64))
    pixels = np.float32(np.asarray(small_image).reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant_colors_hvs = []
    dominant_colors_rgb = []
    for i in range(0, n_colors):
        color = palette[i]
        h, s, v = colorsys.rgb_to_hsv(int(color[0]), int(color[1]), int(color[2]))
        dominant_colors_hvs.append((h, s, v, counts[i]))
        dominant_colors_rgb.append(((int(color[0]), int(color[1]), int(color[2])), int(counts[i])))

    dominant_colors_rgb.sort(key=lambda x: x[1], reverse=True)
    dominant_colors_hvs.sort(key=lambda x: x[1], reverse=True)
    ordered_colors_hvs = []
    for color in dominant_colors_hvs:
        color_dict = {"Hue": color[0],
                      "Saturation": color[1],
                      "Value": color[2],
                      "Count": color[3]
                      }
        ordered_colors_hvs.append(color_dict)
    return ordered_colors_hvs, dominant_colors_rgb


def get_length(filename):
    # To Do: Replace with open-cv library
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def main():
    n_colors = 5
    start_time = time.time()
    video_folders = "temp_clips"
    video_dirs = [f for f in os.listdir(video_folders) if os.path.isdir(os.path.join(video_folders, f))]
    print(f'Found video paths: {video_dirs}')
    output_path = 'clip_data'

    # video_dirs = video_dirs[:10]
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
        data_output_path = os.path.join(output_path, video_dir + "-data.json")
        video_color_data = []
        with io.open(data_output_path, "w", encoding="utf-8") as f:
            f.write('{ "video_path" : "' + video_dir + '",\n')
            f.write('"clip_data" : [\n')
            for clip_path in clip_paths:
                j += 1
                if j % 10 == 0:
                    print(f'Analyzing video {i} of {video_count}, clip {j} of {clip_count}')

                full_clip_path = os.path.join(video_folders, video_dir, clip_path)
                clip_data = dict()
                clip_data["fullpath"] = full_clip_path
                clip_data["length"] = get_length(filename=full_clip_path)
                clip_data["dominant colors"], scene_ordered_colors_rgb = get_dominant_colors(full_clip_path, n_colors)
                video_color_data.append(scene_ordered_colors_rgb)
                f.write('\t' + str(clip_data).replace("'", '"') + ',\n')
            f.write(']}')

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
            print(f'Time since start {time.time()-start_time}')


if __name__ == "__main__":
    main()
