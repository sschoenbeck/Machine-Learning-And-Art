"""
This files takes in uncut videos and exports clips to temp_clips.
"""

import os
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


def main():
    print(f'ffmepeg is available: {video_splitter.is_ffmpeg_available()}')
    print(f'mkvmerge is available: {video_splitter.is_mkvmerge_available()}')

    video_folder = "full_videos"
    video_paths = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
    print(f'Found video paths: {video_paths}')

    output_file_template = "temp_clips\$VIDEO_NAME\Scene$SCENE_NUMBER.mp4"
    video_count = len(video_paths)
    i = 0
    for video_path in video_paths:
        i += 1
        video_name = video_path.split('.')[0]
        print(f'Cutting video {i} of {video_count}: {video_name}')
        current_video_path = os.path.join(video_folder, video_path)
        found_scenes = find_scenes(current_video_path)
        try:
            video_splitter.split_video_mkvmerge(input_video_paths=[current_video_path], scene_list=found_scenes,
                                                output_file_template=output_file_template,
                                                video_name=video_name, suppress_output=False)
        except:
            pass


if __name__ == "__main__":
    main()
