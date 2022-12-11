#command to start docker: docker run -it --rm --gpus all --device /dev/nvidia0  --device /dev/nvidia-modeset --device /dev/nvidiactl -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
#command to run in docker: python /current/main.py OieROrpzYuo.mp4 output.h5

from vsum_tools import *
from vasnet import *
import shutil
import os 
segment_length = 2 #in seconds

sampling_rate = 2 #in frame per second
model_file_path = 'tvsum_splits_1_0.6187176441788906.tar.pth.tar'

model_func, preprocess_func, target_size = model_picker('inceptionv3', 'max')
import h5py

import cv2
import sys


import time 

start = time.time()
total_time = 0
# %rm test.h5
# new_h5 = h5py.File('test.h5', 'w')

video_path = f"/current/{sys.argv[1]}"
# output_path = f"/current/{sys.argv[2]}"

video_basename = os.path.basename(video_path) 

video = cv2.VideoCapture(video_path)


got, frame = video.read()
print(got)

fps = (video.get(cv2.CAP_PROP_FPS))
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
padding = lambda i: '0'*(6-len(str(i))) + str(i)

print(fps, frameCount, "(width*height)=", size)

changepoints = [(i, i+ int(segment_length*fps)-1) for i in range(0, int(frameCount), int(segment_length*fps))]
nfps = [x[1] - x[0] + 1 for x in changepoints]

picks = []
i = 0
while True:
	picks += [int(fps*i + x* int(fps//sampling_rate)) for x in range(sampling_rate)]
	i += 1
	if picks[-1] > frameCount :
		break

picks = [i for i in picks if i < frameCount]


tmp_total = time.time() - start
print(f"---Caculating picks and change points for 2s segments took {tmp_total - total_time} second(s)")
total_time = tmp_total




frames, features = extract_frame_and_features4video(model_func, preprocess_func, target_size, picks, video_path)

tmp_total = time.time() - start
print(f"---Sampling frames, saving sampled frame and extract feature took {tmp_total - total_time} second(s)")
total_time = tmp_total
# frames2 = load_saved_frame(save_frame_dir)

# cv2.imwrite('loaded.png', frames2[0])
# cv2.imwrite('extracted.png', frames[0])
# print (len(frames2) == len(frames))

# quit()
# features = extract_features4video(model_func, preprocess_func, target_size, picks, video_path)

# for i in frames:
	

# print(changepoints, nfps, picks, features, sep = '\n')



aonet = AONet()

aonet.initialize(f_len = len(features[0]))
aonet.load_model(model_file_path)
print("load model successfull")
predict = aonet.eval(features)
# print(p)
threshold = 0.5


summary = generate_summary(predict, np.array(changepoints), int(frameCount),nfps, picks)

tmp_total = time.time() - start
print(f"---Predicting important score and calculate summary took {tmp_total - total_time} second(s)")
total_time = tmp_total
# print(summary)
# print(sum(summary), len(summary))


# print("The following frames will be selected into summary ", [i for i in range(len(summary)) if (summary[i] == 1) ])

# sum_video_name = 'video.mp4'


sum_video_path = f"/current/{video_basename}_output/"
sampled_frame_dir = f"{sum_video_path}/save_frame"


try:
	
	shutil.rmtree(sum_video_path)
except:
	pass




os.mkdir(sum_video_path)
os.mkdir(sampled_frame_dir)

for i in range(len(frames))	:
	cv2.imwrite(f"{sampled_frame_dir}/frame_pos_{picks[i]}.png", frames[i])

new_h5 = h5py.File(f"{sum_video_path}/metadata.h5", 'w')
new_h5.create_dataset('changepoints', data=changepoints)
new_h5.create_dataset('predict', data=predict)
new_h5.create_dataset('picks', data=picks)
new_h5.create_dataset('nfs', data=nfps)
new_h5.create_dataset('generated_summary', data=summary)
new_h5.close()

generate_summary_video(video_path, sum_video_path, summary)


tmp_total = time.time() - start
print(f"---Extracting frames and stitching them into summary video took {tmp_total - total_time} second(s)")
total_time = tmp_total