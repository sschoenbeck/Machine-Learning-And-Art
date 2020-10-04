"""
This files takes in uncut videos and returns an array of clips and exports clips to temp_clips.
"""
import io
import os
import subprocess


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def main():
    video_folders = "temp_clips"
    video_dirs = [f for f in os.listdir(video_folders) if os.path.isdir(os.path.join(video_folders, f))]
    print(f'Found video paths: {video_dirs}')

    output_path = 'clip_data'

    video_count = len(video_dirs)
    i = 0
    for video_dir in video_dirs:
        i += 1
        clip_paths = [f for f in os.listdir(os.path.join(video_folders, video_dir)) if
                      os.path.isfile(os.path.join(video_folders, video_dir, f))]
        video_data = []
        clip_count = len(clip_paths)
        j = 0
        data_output_path = os.path.join(output_path, video_dir + "-data.txt")
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
                video_data.append(clip_data)
                f.write(f'\t{clip_data},\n')
            f.write('}')


main()
