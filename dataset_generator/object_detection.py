import os
import subprocess

root_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/'
img_dir = 'temp/imgs'
img_paths = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
img_text_path = 'temp/img_paths.txt'

f = open(img_text_path, 'w')
for img_path in img_paths:
    f.write(f'{root_path}{img_dir}/{img_path} \n')
    print(root_path + img_dir + '/' + img_path)

f.close()


result_path = 'C:/Users/Simon/git/Machine-Learning-And-Art/dataset_generator/temp/results.json'

result = subprocess.run(['darknet.exe', 'detector', 'test', 'cfg/coco.data', 'cfg/yolov4.cfg', 'yolov4.weights',
                         '-ext_output', '-dont_show', '-out', result_path, '<', f'{root_path}{img_text_path}'],
                        cwd="C:/Users/Simon/git/darknet/build/darknet/x64",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)

result_str = str(result)
result_list = result_str.split('\\r\\n')
for line in result_list:
    print(line)
