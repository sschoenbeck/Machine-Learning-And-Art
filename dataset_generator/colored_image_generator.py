import PIL
from PIL import Image
import numpy as np
import pandas as pd
import random


def analyze_colors(pixels):
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
                    rgb_cube[f'({r_point}, {g_point}, {b_point})'] += r_distances[r_point] + g_distances[g_point] + \
                                                                      b_distances[b_point]
    return rgb_cube


def generate_color_array(color_ranges, length):
    red_range, green_range, blue_range = color_ranges
    pixels = []
    for i in range(0, length):
        r = max(min(random.randrange(red_range[0], red_range[1]), 255), 0)
        g = max(min(random.randrange(green_range[0], green_range[1]), 255), 0)
        b = max(min(random.randrange(blue_range[0], blue_range[1]), 255), 0)
        pixels.append((r, g, b))
    return pixels


def tuple_to_int(pixels):
    image_data = []
    for pixel in pixels:
        r, g, b = pixel
        image_data.append(r + 256 * g + 65536 * b)
    return image_data


def create_data_dict_from_array(fullpath, pixels, m_color, s_color):
    clip_data = analyze_colors(pixels)
    clip_data["fullpath"] = fullpath
    clip_data["length"] = 1.0
    if m_color == s_color:
        clip_data["color"] = m_color
    else:
        clip_data["color"] = f'{m_color} and {s_color}'
    return clip_data


def create_image(path, pixels):
    image_data = tuple_to_int(pixels)
    img = Image.new('RGB', (64, 64))
    img.putdata(image_data)
    image_path = f'noisy_images/{path}.png'
    img.save(image_path)


def main():
    df = pd.DataFrame()
    duplicates = 1
    colors = {'Black': ((0, 63), (0, 63), (0, 63)),
              'Navy': ((0, 63), (0, 63), (64, 191)),
              'Blue': ((0, 63), (0, 63), (192, 255)),
              'Green': ((0, 63), (64, 191), (0, 63)),
              'Teal': ((0, 63), (64, 191), (64, 191)),
              'DeepSkyBlue': ((0, 63), (64, 191), (192, 255)),
              'Lime': ((0, 63), (192, 255), (0, 63)),
              'SpringGreen': ((0, 63), (192, 255), (64, 191)),
              'Cyan': ((0, 63), (192, 255), (192, 255)),
              'DarkRed': ((64, 191), (0, 63), (0, 63)),
              'Purple': ((64, 191), (0, 63), (64, 191)),
              'Violet': ((64, 191), (0, 63), (192, 255)),
              'Olive': ((64, 191), (64, 191), (0, 63)),
              'Gray': ((64, 191), (64, 191), (64, 191)),
              'Malibu': ((64, 191), (64, 191), (192, 255)),
              'Chartreuse': ((64, 191), (192, 255), (0, 63)),
              'MintGreen': ((64, 191), (192, 255), (64, 191)),
              'Anakiwa': ((64, 191), (192, 255), (192, 255)),
              'Red': ((192, 255), (0, 63), (0, 63)),
              'Rose': ((192, 255), (0, 63), (64, 191)),
              'Fuchsia': ((192, 255), (0, 63), (192, 255)),
              'Orange': ((192, 255), (64, 191), (0, 63)),
              'LightCoral': ((192, 255), (64, 191), (64, 191)),
              'Pink': ((192, 255), (64, 191), (192, 255)),
              'Yellow': ((192, 255), (192, 255), (0, 63)),
              'Dolly': ((192, 255), (192, 255), (64, 191)),
              'White': ((192, 255), (192, 255), (192, 255))
              }
    for m_color in colors:
        for s_color in colors:
            for i in range(0, duplicates):
                if m_color == s_color:
                    for m_str in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                        pixels = []
                        pixels += generate_color_array(colors[m_color], int(round(m_str * 4096)))
                        pixels += generate_color_array(((0, 255), (0, 255), (0, 255)), 4096 - len(pixels))
                        path = f'noisy_image_{m_color}_{m_str}_{s_color}_{0.0}_{i}'
                        data_dict = create_data_dict_from_array(path, pixels, m_color, s_color)
                        df = df.append(data_dict, ignore_index=True)
                        # create_image(path, pixels)
                        print(f'Finished {path}')
                else:
                    for m_str in range(2, 10):
                        s_max = min(10 - m_str, m_str)
                        for s_str in range(2, s_max):
                            pixels = []
                            pixels += generate_color_array(colors[m_color], int(round(m_str * 409.6)))
                            pixels += generate_color_array(colors[s_color], int(round(s_str * 409.6)))
                            pixels += generate_color_array(((0, 255), (0, 255), (0, 255)), 4096 - len(pixels))
                            path = f'noisy_image_{m_color}_{m_str}_{s_color}_{s_str}_{i}'
                            data_dict = create_data_dict_from_array(path, pixels, m_color, s_color)
                            df = df.append(data_dict, ignore_index=True)
                            # create_image(path, pixels)
                            print(f'Finished {path}')

    df.to_csv(f'noisy_images/created_colored_images.csv', index=False)


main()
