from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm


def get_dominant_colors(n_colors, img_path):
    with Image.open(img_path) as im:
        small_image = im.resize((64, 64))
    pixels = np.float32(np.asarray(small_image).reshape(-1, 3))

    dominant_colors_rgb = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    for i in range(0, n_colors):
        color = palette[i]
        dominant_colors_rgb.append(((int(color[0]), int(color[1]), int(color[2])), int(counts[i])))

    dominant_colors_rgb.sort(key=lambda x: x[1], reverse=True)

    return dominant_colors_rgb


def main():
    working_dir = os.getcwd()
    n_colors = 5
    img_folder = os.path.join(working_dir, "temp_imgs\\movie")
    output_path = "timeline_color.png"
    img_paths = [f for f in os.listdir(img_folder) if
                 os.path.isfile(os.path.join(img_folder, f))]
    img_width = 4096
    img_height = 32 * len(img_paths)
    video_color_list = []
    #img_paths = img_paths[:5]
    for i in tqdm(range(len(img_paths)),
                  desc="Analyzing Clips",
                  ascii=False, ncols=75):
        img_path = img_paths[i]
        dominant_colors = get_dominant_colors(n_colors, os.path.join(img_folder, img_path))
        clip_color_list = []
        for color_data in dominant_colors:
            for x in range(0, color_data[1]):
                clip_color_list.append(color_data[0])
        for y in range(0, 32):
            video_color_list += clip_color_list

    img = Image.new('RGB', (img_width, ))
    img.putdata(video_color_list)
    imgR.save(output_path)


if __name__ == "__main__":
    main()
