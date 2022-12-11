#command to start docker: docker run -it --rm --gpus all --device /dev/nvidia0  --device /dev/nvidia-modeset --device /dev/nvidiactl -u $(id -u):$(id -g) -v `pwd`:/current truongan/uit-vsum:1.0 bash
#command to run in docker: python /current/main.py OieROrpzYuo.mp4 output.h5

from vsum_tools import *
from vasnet_add_LSTM_with_training import *
# from vasnet_with_training import *
import shutil
import os 

import h5py

import sys


import time 

import torch
# from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init



def train(hps, f_len = 2048):
    os.makedirs(hps.output_dir, exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits
    f = open(hps.output_dir + '/results.txt', 'wt')

    for split_filename in hps.splits:
        dataset_name, dataset_type, splits = parse_splits_filename(split_filename)
        
        # print("---------")
        # print(parse_splits_filename(split_filename))
        # print("-----------------")

        # For no augmentation use only a dataset corresponding to the split file
        datasets = None
        if dataset_type == '':
            datasets = hps.get_dataset_by_name(dataset_name)

        if datasets is None:
            datasets = hps.datasets

        f_avg = 0
        n_folds = len(splits)
        for split_id in range(n_folds):
            ao = AONet(hps)
            ao.load_datasets(datasets=datasets)

            ao.initialize(f_len = f_len) # AN EDIT : pass f_len through
            
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(split_id=split_id)

            fscore, fscore_epoch = ao.train(output_dir=hps.output_dir)
            f_avg += fscore

            # Log F-score for this split_id
            f.write(split_filename + ', ' + str(split_id) + ', ' + str(fscore) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename)
            log_dir, _ = os.path.splitext(log_file)
            log_dir += '_' + str(split_id)
            log_file = os.path.join(hps.output_dir, 'models', log_dir) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, 'models', ), exist_ok=True)
            os.system('mv ' + hps.output_dir + '/models_temp/' + log_dir + '/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
            os.system('rm -rf ' + hps.output_dir + '/models_temp/' + log_dir)

            print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))

        # Write average F-score for all splits to the results.txt file
        f_avg /= n_folds
        f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + '\n')
        f.flush()

    f.close()


import sys

import os
#AN's path
parser = argparse.ArgumentParser("PyTorch implementation of paper \"Summarizing Videos with Attention\"")
parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
parser.add_argument('-t', '--train', action='store_true', help="Train")
parser.add_argument('-c', '--use_cuda', action='store_true', default=False, help="Use CUDA")
parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")

import timeit
import datetime 
import sys
import h5py


from torch.autograd import Variable



def train_wrapper(split_file):
  from datetime import datetime
  sys.argv = "main.py "
  sys.argv += '-c '
#   sys.argv += '-d ' + 'for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg.h5,for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg.h5'
  # sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5'
#   sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/ucf_crime_anomaly_test_poissonpmf_score_0.2-0.3_videos_inceptionv3_avg.h5'
#   sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg_poissonpmf_shortened.h5'
  # sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg_shortened_longer.h5'
  # sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg_gaussian-pdf_shorten.h5'
  # sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/ucf_crime_anomaly_test_videos_inceptionv3_avg_0-1score_shortened.h5'
  # sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/HACS_segment_first_0-100pntruongan2005_inceptionv3_avg_shorten_3_event_max.h5'
  # sys.argv += '-d ' + 'for.training/eccv16_dataset_ovp_google_pool5.h5,for.training/eccv16_dataset_summe_google_pool5.h5,for.training/eccv16_dataset_tvsum_google_pool5.h5,for.training/eccv16_dataset_youtube_google_pool5.h5'
  sys.argv += '-d ' + 'for.training/merged.tvsum.summe-inceptionv3_avg.h5.h5,for.training/HACS_segment_first_0-100pntruongan2005_inceptionv3_avg_shorten_3_event_max.h5,for.training/HACS_segment_first_100-200pntruongan2005_inceptionv3_avg_shorten_3_event_max.h5,for.training/HACS_segment_first_200-300pntruongan2005_inceptionv3_avg_shorten_3_event_max.h5'
  sys.argv += '  -o ' + f"for.training/vasnet_add_LSTM_retrain_{os.path.basename(split_file)}-{str(datetime.now()).replace(' ', 'T')}/"
  sys.argv += '  -s ' + split_file
  sys.argv = sys.argv.split()

  args = parser.parse_args()


  # MAIN
  #======================
  hps = HParameters()
  hps.load_from_args(args.__dict__)


  #Get feature length
  h5 = h5py.File(hps.datasets[2], 'r');
  f_len =(h5[list(h5.keys())[0]]['features'].shape[1])

  train(hps, f_len)

train_wrapper('for.training/summe_canonical_inceptionv3_splits.json')
# train_wrapper('for.training/tvsum_augmentation_with_summe_hacs_0-100_gaussian-pdf_splits.json')
# train_wrapper('for.training/tvsum_augmentation_with_summe_hacs_0-300_gaussian-pdf_splits.json')