import subprocess

file1_path = 'out1.mp4'
file2_path = 'out2.mp4'
output_file_path = 'concat_out.mp4'
source_path = 'E:/UCARE/Movies/2001_A_SPACE_ODYSSEY.mp4'
start_time = 0
clip_length = '02:58.2'

result = subprocess.run(f'ffmpeg -y -ss {start_time} -i {source_path} -t {clip_length} -c copy {file2_path}',
                        cwd="C:/Users/Simon/git/Machine-Learning-And-Art/timeline_generation/temp",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
print(result)

clip_data = [{'start': 0, 'clip_length': '02:58.2'},
             {'start': 200, 'clip_length': '02:58.2'},
             {'start': 400, 'clip_length': '02:58.2'}]

if file2_path[:-4] == '.mp4':
    txt_file_path = 'list.txt'
    f = open(txt_file_path, '')
    f.write(f'{file1_path}\n{file2_path}')
    f.close()

    result = subprocess.run(f'ffmpeg -y -f concat -i {txt_file_path} -c {output_file_path}',
                            cwd="C:/Users/Simon/git/Machine-Learning-And-Art/timeline_generation/temp",
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    print(result)
else:
    result = subprocess.run(
        f'ffmpeg -y -i {file1_path} -i {file2_path} -filter_complex "[0:v][0:a][1:v][1:a] concat=n=2:v=1:a=1'
        f' [outv] [outa]" -map "[outv]" -map "[outa]" {output_file_path}',
        cwd="C:/Users/Simon/git/Machine-Learning-And-Art/timeline_generation/temp",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    print(result)
