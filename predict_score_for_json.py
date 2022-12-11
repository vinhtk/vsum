#command to start docker: docker run -it --rm --gpus all --device /dev/nvidia0  --device /dev/nvidia-modeset --device /dev/nvidiactl -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
#command to run in docker: python /current/main.py OieROrpzYuo.mp4 output.h5

import sys

from vsum_tools import *
from vasnet import *
import shutil
import os 
segment_length = 2 #in seconds

sampling_rate = 2 #in frame per second
model_root_dir = '/working'
model_file_path = model_root_dir + '/model.pth.tar'



def generate_json(video_path):  
  import cv2
  import base64
  video = cv2.VideoCapture(video_path)

  segment_length = 2
  sampling_rate = 2

  got, frame = video.read()
  if ( not got):
    print(f"Cant read video from {video_path}", file=sys.stderr)
    return

  fps = (video.get(cv2.CAP_PROP_FPS))
  frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
  size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  # size_param = "%dx%d"%size 
  # padding = lambda i: '0'*(6-len(str(i))) + str(i)

  print(fps, frameCount, size)

  changepoints = [(i, i+ int(segment_length*fps)-1) for i in range(0, int(frameCount), int(segment_length*fps))]
  picks = []
  i = 0
  while True:
    picks += [int(fps*i + x* int(fps//sampling_rate)) for x in range(sampling_rate)]
    i += 1
    if picks[-1] > frameCount :
      break

  picks = [i for i in picks if i < frameCount]

  import numpy as np

  writer = None
  (W, H) = (None, None)

  features = []
  pick_frames = []


  output = dict(    
    {"Service": ["Vsum"],
      "Split": True,
      "FrameWidth": size[0],
      "FrameHeight": size[1],
      "Attributes": {
        "Roi":[0,0,1920,1080],  # khung ROI quan tâm
        "From": 1606988601,  # Thời gian bắt đầu
        "To": 1606988611,  # Thời gian kết thúc  (nếu chỉ 1 ảnh thì thời gian From = To)
        "CapturedGroup": "",  # Thông tin dùng để nhóm các Camera (có thể bỏ qua)
        "CapturedBy": "PTZ Ly Thai To",  # Tên camera 
        "Tag": {  # Thông tin để lưu trữ vào storage ảnh (các này không cần thay đổi)
          "Tag": "VVMS_Face",
          "Ext": ""
        },
        "Longitude": 106.65261268615724,  #  Toạ độ longitude
        "Latitude": 10.79241719694889,  #  Toạ độ latitude
        "Altitude": 0.0,  #  Toạ độ Altitude
      },
      "Images": []
    }  
  )

  

  n_frames = -1

  total = len(picks)
  count = 0

  selected_frame = picks[count]

  while True:
	  # read the next frame from the file
    (grabbed, frame) = video.read()
	  # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
      break

    n_frames += 1

    #if (n_frames in picks): # banana code - làm chậm 10 lần
    if( selected_frame == n_frames):
      count += 1
      if(count < total):
        selected_frame = picks[count]
      if(count % 50 == 0):
        print("-- Processing frame ", count, "/", total, ": ", n_frames)
      img_array = cv2.imencode('.jpg', frame)[1].tobytes()
      output['Images'].append(dict({
        'Guid' :n_frames,
        'Image': dict({
          'Data': base64.b64encode(img_array).decode(),
          "Width" : size[0],
          "Height": size[1]
        })
      }))
  import json
  # print(json.dump(output,open('sample_output.json','w'), indent=2))  
  return json.dumps(output, indent=2)

def extract_features_from_images(model_func, preprocess_func, target_size, images):
  import cv2
  from tensorflow.keras.preprocessing import image
  import numpy as np
  import base64
  features = []
  converted_frame = []
  count = 0
  for frame in images:
    # print(base64.b64decode(frame['Image']['Data']))
    a = bytearray(base64.b64decode(frame['Image']['Data']))
    # break
    frame = cv2.imdecode(np.asarray(a), cv2.IMREAD_UNCHANGED)
    count += 1
    if(count % 50 == 0):
      print("-- Processing frame ", count, "/", len(images))
    # convert it from BGR to RGB
    # ordering, resize the frame to a fixed target_size, and then
    # perform mean subtraction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size).astype("float32")
    img_array = image.img_to_array(frame)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_func(expanded_img_array)
    converted_frame.append(preprocessed_img)

    # print(n_frames)
    # debug only
    #if(count>10):
    #  break

  converted_frame = np.vstack(converted_frame)
  print("pick frames ", len(converted_frame))
  # xử lý theo batch
  print("-- Extracting features: ...")
  features = model_func.predict(converted_frame, batch_size = 32)
  print("-- Done. Feature size: ", features.shape)

  return features

import argparse

parser = argparse.ArgumentParser(description='Generate json from video or read json file to predict segment important score')

parser.add_argument('--save_json','-e', dest='save_json_file', 
                    help='Extract frame from the video and save it into a json file. If this argument is given, the file_name argument is expected to be a video file', required=False)
parser.add_argument('--save_score','-c', dest='save_score_file', 
                    help='Read extract frame from json file, predict each frame important score and save those score in another json file. This is default action when save_json argument is not given', required=False)
parser.add_argument('file_name', metavar='file_name', type=str, 
                    help='This is either a video file or json file, depend on other options')
# parser.add_argument('--in', '-i', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='The input video (if out argument is NOT empty) or the input video path if --out is missing')

argument = parser.parse_args()
print(argument, file=sys.stderr)

if (argument.save_json_file != None):
  a = generate_json(argument.file_name)
  with open(argument.save_json_file, 'w') as f:
    f.write(a)

else:
  model_func, preprocess_func, target_size = model_picker('inceptionv3', 'max')
  import json
  with open(argument.file_name, 'r') as f:  inp = json.load(f)
  import h5py

  import cv2
  import sys
  features = extract_features_from_images(model_func, preprocess_func, target_size, inp['Images'])
  
  aonet = AONet()

  aonet.initialize(f_len = len(features[0]))
  aonet.load_model(model_file_path)
  print("load model successfull")
  predict = aonet.eval(features)

  try:
    with open(argument.save_score_file, 'w') as f:
      print(predict, file=f)
  except:
    print(predict)