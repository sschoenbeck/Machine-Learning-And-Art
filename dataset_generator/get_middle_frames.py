import os
import cv2
from PIL import Image
import time


def save_frame(full_clip_path, video_dir):
    clip_name = full_clip_path.split('\\')[-1].split('.')[0]
    img_path = f'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/middle_frames/{video_dir}/{clip_name}.png'
    cap = cv2.VideoCapture(full_clip_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = int(frame_count/2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, img = cap.read()
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(RGB_img).save(img_path)
    cap.release()


def main():
    video_folders = "temp_clips"
    video_dirs = [f for f in os.listdir(os.path.join(video_folders, video_dir)) if
                      os.path.isfile(os.path.join(video_folders, video_dir, f))]
    # ["C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/temp_clips/2001_A_SPACE_ODYSSEY"]
    video_count = len(video_dirs)
    i = 0
    for video_dir in video_dirs:
        video_name = video_dir.split('/')
        print(f'Analyzing: {video_dir}')
        i += 1
        clip_paths = [f for f in os.listdir(os.path.join(video_folders, video_dir)) if
                      os.path.isfile(os.path.join(video_folders, video_dir, f))]
        clip_count = len(clip_paths)
        j = 0
        for clip_path in clip_paths:
            j += 1
            if j % 10 == 0:
                print(f'Analyzing video {i} of {video_count}, clip {j} of {clip_count}')
            full_clip_path = os.path.join(video_folders, video_dir, clip_path)
            save_frame(full_clip_path, video_dir)


if __name__ == "__main__":
    main()
