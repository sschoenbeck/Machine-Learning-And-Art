import pandas as pd
import os
import cv2
import subprocess
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from sklearn import preprocessing
from scenedetect import VideoManager, SceneManager, video_splitter
from scenedetect.detectors import ContentDetector
import tensorflow as tf
from tensorflow.keras import layers
import random
import time


def find_scenes(video_path, threshold=30.0):
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
                    rgb_cube[f'({r_point}, {g_point}, {b_point})'] += r_distances[r_point] + \
                                                                      g_distances[g_point] + b_distances[b_point]
    return rgb_cube


def detect_objects(img_text_path, result_path, darknet_path):
    object_id_max = 80
    print('Detecting Objects')
    _ = subprocess.run(['darknet.exe', 'detector', 'test', 'cfg\\coco.data', 'cfg\\yolov4.cfg', 'yolov4.weights',
                        '-ext_output', '-dont_show', '-out', result_path, '<', img_text_path],
                       cwd=darknet_path,
                       shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    results_json = json.loads(open(result_path).read().replace('\\', '\\\\'))
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


def process_movie(template_movie_path, darknet_path, generator_type):
    working_dir = os.getcwd()
    print(f'Work directory: {working_dir}')

    video_name = template_movie_path.split('\\')[-1].split('.')[0]
    print(f'Cutting Video: {video_name}')
    found_scenes = find_scenes(template_movie_path)
    movie_df = pd.DataFrame()

    img_text_path = working_dir + f'\\temp_imgs\\movie\\image_list.txt'
    result_path = working_dir + f'\\data\\movie_results.json'

    f = open(img_text_path, "w")
    for i in tqdm(range(len(found_scenes)),
                  desc="Analyzing Scenes",
                  ascii=False, ncols=75):
        scene = found_scenes[i]
        scene_dict = dict()
        start_timecode = scene[0]
        end_timecode = scene[1]

        scene_dict['start_time'] = start_timecode.get_seconds()
        scene_dict['start_frame'] = int(start_timecode.get_frames())
        scene_dict['end_time'] = end_timecode.get_seconds()
        scene_dict['end_frame'] = int(end_timecode.get_frames())
        scene_dict['length'] = end_timecode - start_timecode
        scene_dict['middle_frame'] = int((start_timecode.get_frames() + end_timecode.get_frames()) / 2)
        scene_dict['img_path'] = f'temp_imgs\\movie\\{str(i).rjust(4, "0")}.png'
        scene_dict['video_path'] = template_movie_path

        save_frame(template_movie_path, scene_dict['img_path'], scene_dict['middle_frame'])
        rgb_cube = get_dominant_colors(scene_dict['img_path'])
        scene_dict.update(rgb_cube)
        movie_df = movie_df.append(scene_dict, ignore_index=True)

        img_path = working_dir + '\\' + scene_dict['img_path'] + '\n'
        f.write(img_path)

    f.close()
    movie_df.to_csv(f'data\\movie_data.csv', index=False)
    object_df = detect_objects(img_text_path, result_path, darknet_path)
    complete_df = pd.concat([movie_df, object_df], axis=1)

    column_labels = complete_df.columns
    if generator_type == 'movie_full_clips' or generator_type == 'movie_edited_clips':
        padding = pd.DataFrame(np.zeros((5, len(column_labels))), columns=column_labels)
        complete_df = pd.concat([padding, complete_df])
    complete_df.to_csv(f'data\\complete_movie_data.csv', index=False)
    return complete_df


def process_clips(sample_clip_dir, darknet_path):
    working_dir = os.getcwd()
    print(f'Work directory: {working_dir}')
    clip_paths = [f for f in os.listdir(sample_clip_dir) if os.path.isfile(os.path.join(sample_clip_dir, f))]

    img_text_path = working_dir + f'\\temp_imgs\\clips\\image_list.txt'
    result_path = working_dir + f'\\data\\clips_results.json'
    f = open(img_text_path, "w")

    clip_df = pd.DataFrame()
    for i in tqdm(range(len(clip_paths)),
                  desc="Analyzing Clips",
                  ascii=False, ncols=75):
        clip_path = clip_paths[i]
        clip_name = clip_path.split('.')[0]
        full_clip_path = os.path.join(sample_clip_dir, clip_path)

        clip_dict = dict()
        cap = cv2.VideoCapture(full_clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        clip_dict['start_time'] = 0
        clip_dict['start_frame'] = 0
        clip_dict['end_time'] = duration
        clip_dict['end_frame'] = frame_count
        clip_dict['length'] = duration
        clip_dict['middle_frame'] = int(frame_count / 2)
        clip_dict['img_path'] = f'temp_imgs\\clips\\{clip_name}.png'
        clip_dict['video_path'] = full_clip_path

        save_frame(full_clip_path, clip_dict['img_path'], clip_dict['middle_frame'])
        rgb_cube = get_dominant_colors(clip_dict['img_path'])
        clip_dict.update(rgb_cube)
        clip_df = clip_df.append(clip_dict, ignore_index=True)

        img_path = working_dir + '\\' + clip_dict['img_path'] + '\n'
        f.write(img_path)

    f.close()
    clip_df.to_csv(f'data\\clip_data.csv', index=False)
    object_df = detect_objects(img_text_path, result_path, darknet_path)
    complete_df = pd.concat([clip_df, object_df], axis=1)
    complete_df.to_csv(f'data\\complete_clip_data.csv', index=False)
    return complete_df


def distance_calculator(expected_frame, possible_frame):
    color_difference = 0
    for i in range(27):
        color_difference += abs(expected_frame[i] - possible_frame[i])
    return color_difference


def best_frame_finder(expected_frame, possible_frames_df, used_clips, best_frames=1, repeat_penalty=500):
    possible_frame_matrix = possible_frames_df.to_numpy()
    possible_frame_cont = len(possible_frame_matrix)
    frame_scores = []
    for i in range(possible_frame_cont):
        if i in used_clips:
            frame_scores.append(
                [i, distance_calculator(expected_frame, possible_frame_matrix[i]) + repeat_penalty * used_clips[i]])
        else:
            frame_scores.append([i, distance_calculator(expected_frame, possible_frame_matrix[i])])

    frame_scores = sorted(frame_scores, key=lambda x: x[1])
    if best_frames < len(possible_frames_df):
        frame_scores = frame_scores[0:best_frames]
    if frame_scores[0][0] in used_clips:
        used_clips[frame_scores[0][0]] = used_clips[frame_scores[0][0]] + 1
    else:
        used_clips[frame_scores[0][0]] = 1
    return frame_scores, used_clips


def create_dataset(X, time_steps=5):
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        X_train.append(v)
        Y_train.append(X.iloc[i + time_steps].to_numpy())
        if random.random() * 10 > 9:
            X_test.append(v)
            Y_test.append(X.iloc[i + time_steps].to_numpy())
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def create_model(numerical_df):
    x = numerical_df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_df = pd.DataFrame(x_scaled, columns=numerical_df.columns)
    scaled_df.plot()

    column_indices = {name: i for i, name in enumerate(scaled_df.columns)}

    num_features = scaled_df.shape[1]
    TIME_STEPS = 5

    X_train, y_train, X_test, y_test = create_dataset(scaled_df, time_steps=TIME_STEPS)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print("input shape", X_train[0][0].shape, X_train[0][1].shape)
    model = tf.keras.Sequential()
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                units=128,
                input_shape=(X_train[0][0], X_train[0][1])
            )
        )
    )
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(units=y_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        shuffle=False
    )


def generate_timeline_from_template(template_data_df, sample_data_df, frame_rate):
    working_dir = os.getcwd()
    clip_location_col = 34
    sample_data_matrix = sample_data_df.to_numpy()

    used_clips = dict()
    template_data_list = template_data_df.values.tolist()
    for i in tqdm(range(len(template_data_list)),
                  desc="Generating Timeline",
                  ascii=False, ncols=75):
        expected_frame = template_data_list[i]
        clip_length = expected_frame[30]
        best_clip, used_clips = best_frame_finder(expected_frame, sample_data_df, used_clips)
        clip_path = sample_data_matrix[best_clip[0][0]][clip_location_col]
        img_path = sample_data_matrix[best_clip[0][0]][clip_location_col + 1]
        os.system(f'copy {img_path} {working_dir}\\temp_imgs\\timeline\\{str(i).rjust(4, "0")}.png')
        output_path = f'{working_dir}\\timeline\\{str(i).rjust(4, "0")}.MTS'
        subprocess.run(
            f'ffmpeg -y -i \"{clip_path}\" -map 0:v -c:v copy -bsf:v hevc_mp4toannexb {working_dir}\\data\\raw.h265',
            cwd=working_dir,
            shell=True)
        subprocess.run(
            f'ffmpeg -fflags +genpts -r {frame_rate} -i raw.h264 -c:v copy {working_dir}\\data\\temp-video.MTS',
            cwd=working_dir,
            shell=True)
        cmd = f'ffmpeg -y -ss 0 -i \"{clip_path}\" -c copy -t {clip_length} \"{output_path}\"'

        _ = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def generate_timeline_from_full_clips(template_data_df, sample_data_df):
    pass
    numerical_df = template_data_df[
        ["(0, 0, 0)", "(0, 0, 128)", "(0, 0, 255)", "(0, 128, 0)", "(0, 128, 128)", "(0, 128, 255)", "(0, 255, 0)",
         "(0, 255, 128)", "(0, 255, 255)", "(128, 0, 0)", "(128, 0, 128)", "(128, 0, 255)", "(128, 128, 0)",
         "(128, 128, 128)", "(128, 128, 255)", "(128, 255, 0)", "(128, 255, 128)", "(128, 255, 255)", "(255, 0, 0)",
         "(255, 0, 128)", "(255, 0, 255)", "(255, 128, 0)", "(255, 128, 128)", "(255, 128, 255)", "(255, 255, 0)",
         "(255, 255, 128)", "(255, 255, 255)", "middle_frame"]]


def generate_timeline_from_full_edited_clips(template_data_df, sample_data_df):
    pass


def make_movie(output_file_path, template_movie_path, generator_type):
    working_dir = os.getcwd()
    clip_paths = [f for f in os.listdir('timeline') if os.path.isfile(os.path.join('timeline', f))]
    clip_paths.sort()
    txt_file_path = f'{working_dir}\\data\\timeline.txt'

    f = open(txt_file_path, "w")
    for clip_path in clip_paths:
        f.write(f'file \'{working_dir}\\timeline\\{clip_path}\'\n')
    f.close()

    if generator_type == 'replace_video':
        video_path = f'{working_dir}\\data\\temp-video.MTS'
        subprocess.run(f'ffmpeg -f concat -safe 0 -i "{txt_file_path}" -c copy "{video_path}" -y',
                       cwd=working_dir,
                       shell=True)
        subprocess.run(
            f'ffmpeg -y -i \"{video_path}\" -i \"{template_movie_path}\" -c:v copy -map 0:v:0 -map 1:a:0 {output_file_path}',
            cwd=working_dir,
            shell=True)
    else:
        subprocess.run(f'ffmpeg -y -f concat -safe 0 -i {txt_file_path} -c copy {output_file_path}',
                       cwd=working_dir,
                       shell=True)


def get_framerate(working_dir, template_movie_path):
    result = subprocess.run(
        f'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate \"{template_movie_path}\"',
        cwd=working_dir,
        capture_output=True,
        shell=True)
    print(result)
    ratio = str(result.stdout).replace('b', '').replace('\'', '').replace('r', '').replace('n', '').replace('\\', '')
    ratio_list = ratio.split('/')
    frame_rate = float(ratio_list[0]) / float(ratio_list[1])
    print(f'frame_rate: {frame_rate}')
    return frame_rate


def main():
    generator_type = 'replace_video'
    template_movie_path = 'E:\\UCARE\\Movies\\WIZARD OF OZ (1939).mp4'
    # template_movie_path = 'E:\\UCARE\\Movies\\Congratulations.mp4'
    sample_clip_dir = 'E:\\UCARE\\CLOUD FILM MEDIA\\PROXIES'
    sample_clip_dir = 'C:\\Users\\Simon\\git\\Machine-Learning-And-Art\\dataset_generator\\temp\\temp_clips\\2001_A_SPACE_ODYSSEY'
    darknet_path = 'C:\\Users\\Simon\\vcpkg\\installed\\x64-windows\\tools\\darknet'
    output_file_path = f'E:\\movie_{time.time_ns() % 100000}.mp4'
    working_dir = os.getcwd()

    pd.set_option('display.max_columns', None)
    print(f'ffmepeg is available: {video_splitter.is_ffmpeg_available()}')
    print(f'mkvmerge is available: {video_splitter.is_mkvmerge_available()}')
    frame_rate = get_framerate(working_dir, template_movie_path)
    # template_data_df = process_movie(template_movie_path, darknet_path, generator_type)
    sample_data_df = process_clips(sample_clip_dir, darknet_path)
    template_data_df = pd.read_csv('data\\complete_movie_data.csv')
    # sample_data_df = pd.read_csv('data\\complete_clip_data.csv')
    if generator_type == 'replace_video' or generator_type == 'replace_clips':
        generate_timeline_from_template(template_data_df, sample_data_df, frame_rate)

    elif generator_type == 'movie_full_clips':
        generate_timeline_from_full_clips(template_data_df, sample_data_df)

    elif generator_type == 'movie_edited_clips':
        generate_timeline_from_full_edited_clips(template_data_df, sample_data_df)
    # Generate Movie
    make_movie(output_file_path, template_movie_path, generator_type)


if __name__ == "__main__":
    main()
