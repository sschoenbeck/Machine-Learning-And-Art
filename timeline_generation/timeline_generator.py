import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import subprocess
from sklearn import preprocessing
import random

import tensorflow as tf
from tensorflow.keras import layers


def distance_calculator(expected_frame, possible_frame):
    color_difference = random.randint(0, 10000)
    for i in range(27):
        color_difference += abs(expected_frame[i] - possible_frame[i])
    return color_difference


def best_frame_finder(expected_frame, possible_frames_df, used_clips, best_frames=1, repeat_penalty=500):
    possible_frame_matrix = possible_frames_df.to_numpy()
    possible_frame_cont = len(possible_frame_matrix)
    frame_scores = []
    for i in range(possible_frame_cont):
        if i in used_clips:
            frame_scores.append([i, distance_calculator(expected_frame, possible_frame_matrix[i]) + repeat_penalty * used_clips[i]])
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


def create_dataset(X, time_steps=1):
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        X_train.append(v)
        Y_train.append(X.iloc[i + time_steps].to_numpy())
        if random.random() * 10 > 9:
            X_test.append(v)
            Y_test.append(X.iloc[i + time_steps].to_numpy())
    return np.array(X_train), np.array(Y_train),  np.array(X_test),  np.array(Y_test)


def main():
    path = os.path.dirname(os.path.dirname(os.path.abspath('distance_calculation.ipynb')))
    print(path)
    # template_data_df = pd.read_csv(f'{path}/dataset_generator/movie_timelines/WIZARD OF OZ (1939)_complete_data_padded.csv')
    # template_data_df = pd.read_csv(f'{path}/dataset_generator/movie_timelines/2001_A_SPACE_ODYSSEY_complete_data_padded.csv')
    # template_data_df = pd.read_csv(f'{path}/dataset_generator/movie_timelines/2001_A_SPACE_ODYSSEY_complete_data.csv')
    template_data_df = pd.read_csv(f'{path}/dataset_generator/movie_timelines/Tove Styrke Borderline_complete_data.csv')
    print(template_data_df.head())
    #time.sleep(30)
    sample_data_df = pd.read_csv(f'{path}/dataset_generator/movie_timelines/SD_complete_data.csv')
    # sample_data_df = pd.read_csv(f'{path}/dataset_generator/timeline_data/WIZARD OF OZ (1939)_complete_data.csv')

    """
    for i in range(frame_count):
        data = best_frame_finder(frame_matrix[i], sample_data_df, 5, auto_sort=True)
        new_frames.append([i, sample_data_matrix[data[0][0]][img_location_col]])

    """
    numerical_df = template_data_df[
        ["(0, 0, 0)", "(0, 0, 128)", "(0, 0, 255)", "(0, 128, 0)", "(0, 128, 128)", "(0, 128, 255)", "(0, 255, 0)",
         "(0, 255, 128)", "(0, 255, 255)", "(128, 0, 0)", "(128, 0, 128)", "(128, 0, 255)", "(128, 128, 0)",
         "(128, 128, 128)", "(128, 128, 255)", "(128, 255, 0)", "(128, 255, 128)", "(128, 255, 255)", "(255, 0, 0)",
         "(255, 0, 128)", "(255, 0, 255)", "(255, 128, 0)", "(255, 128, 128)", "(255, 128, 255)", "(255, 255, 0)",
         "(255, 255, 128)", "(255, 255, 255)", "middle_frame"]]
    print(numerical_df)

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
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        shuffle=False
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()

    y_pred = model.predict(X_test)

    plt.plot(y_test[0], marker='.', label='true')
    plt.plot(y_pred[0], marker='.', label='pred')
    plt.legend()

    pred_difference = y_test - y_pred
    pred_df = pd.DataFrame()
    for i in range(len(y_test)):
        sample = {'true_color': y_test[i][0],
                 'pred_color': y_pred[i][0]}
        pred_df = pred_df.append(sample, ignore_index=True)
    pred_df.plot()

    plt.plot(pred_difference)

    print(sample_data_df.head())
    # print(sample_data_df.columns[35])
    column_name = list(template_data_df.columns)
    frame_matrix = template_data_df.to_numpy()
    sample_data_matrix = sample_data_df.to_numpy()
    row_length = len(sample_data_matrix[0])

    new_frames = []
    img_location_col = 29
    frame_count = len(frame_matrix)

    generated_clips = pd.DataFrame
    current_time_steps = X_train[0:1]
    print(current_time_steps.shape)
    print('==============')
    used_clips = dict()
    generated_timeline_list = f'generated_timeline_{time.time_ns()}.txt'
    generated_movie_output = f'generated_movie_{time.time_ns()}.mp4'
    f = open(generated_timeline_list, 'a')
    for i in range(10):
        current_time_steps = np.asarray(current_time_steps).astype('float32')
        predicted_clip = model.predict(current_time_steps)[0]
        best_clip, used_clips = best_frame_finder(predicted_clip, sample_data_df, used_clips)
        new_frame = sample_data_matrix[best_clip[0][0]][img_location_col]
        print(len(new_frames), best_clip)
        clip_name = new_frame.split("\\")[1]
        generated_clip_path = f'file \'E:\\UCARE\\CLOUD FILM MEDIA\\PROXIES\\{clip_name}.mp4\'\n'
        print(generated_clip_path)
        f.write(generated_clip_path)

        input_path = f'{path}\\dataset_generator\\{new_frame}'
        output_path = f'{path}\\timeline_generation\\temp\\imgs\\{str(i).rjust(4, "0")}.png'
        print(output_path, new_frame)
        im = Image.open(input_path)
        im = im.resize((1920, 1080))
        im.save(output_path)
        shutil.copyfile(input_path, output_path)

        next_clip = sample_data_matrix[best_clip[0][0]][:27]
        current_time_steps = np.append(current_time_steps, next_clip)
        current_time_steps = np.append(current_time_steps, predicted_clip[27:])
        current_time_steps = current_time_steps[28:].reshape((1, 5, 28))

    f.close()
    frame_rate = 10
    timestamp = str(time.time()).split(".")[0]
    input_path = f'{path}\\timeline_generation\\temp\\imgs\\%04d.png'
    output_path = f'{path}\\timeline_generation\\temp\\output\\movie_{timestamp}.mp4'
    cmd = f'ffmpeg -r {frame_rate} -f image2 -s 1920x1080 -i \"{input_path}\" -vcodec libx264 -crf 25  -pix_fmt yuv420p \"{output_path}\"'
    print(cmd)
    subprocess.run(cmd, shell=True)

    cmd = f'ffmpeg -f concat -safe 0 -i \"{generated_timeline_list}\" -c copy "{generated_movie_output}\"'
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
