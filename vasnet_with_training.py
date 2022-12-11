__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

import random
import numpy as np

import os
import h5py, json
from vsum_tools import *

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.ones(features))
		self.beta = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SelfAttention(nn.Module):

	def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
		super(SelfAttention, self).__init__()

		self.apperture = apperture
		self.ignore_itself = ignore_itself

		self.m = input_size
		self.output_size = output_size

		self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

		self.drop50 = nn.Dropout(0.5)



	def forward(self, x):
		n = x.shape[0]  # sequence length

		K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
		Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
		V = self.V(x)

		Q *= 0.06
		logits = torch.matmul(Q, K.transpose(1,0))

		if self.ignore_itself:
			# Zero the diagonal activations (a distance of each frame with itself)
			logits[torch.eye(n).byte()] = -float("Inf")

		if self.apperture > 0:
			# Set attention to zero to frames further than +/- apperture from the current one
			onesmask = torch.ones(n, n)
			trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
			logits[trimask == 1] = -float("Inf")

		att_weights_ = nn.functional.softmax(logits, dim=-1)
		weights = self.drop50(att_weights_)
		y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
		y = self.output_linear(y)

		return y, att_weights_



class VASNet(nn.Module):

	def __init__(self, m = 2048):
		super(VASNet, self).__init__()

		# self.m = 1024 # cnn features size
		self.m = m ## AN EDIT 2021.05.27 change features size to inceptionv3 2048
		# self.hidden_size = 1024  ## AN EDIT 2021.05.27 I don't know wtf does this do 

		self.att = SelfAttention(input_size=self.m, output_size=self.m)
		
		## AN EDIT
		#self.ka = nn.Linear(in_features=self.m, out_features=1024)
		#self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
		#self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
		#self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

		self.ka = nn.Linear(in_features=self.m, out_features=self.m)

        # Hình như bỏ 2 hàng này nó cũng chạy thì phải
		# self.kb = nn.Linear(in_features=self.ka.out_features, out_features=self.m)
		# self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.m)

		self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.drop50 = nn.Dropout(0.5)
		self.softmax = nn.Softmax(dim=0)
		self.layer_norm_y = LayerNorm(self.m)
		self.layer_norm_ka = LayerNorm(self.ka.out_features)


	def forward(self, x, seq_len):

		m = x.shape[2] # Feature size

		# Place the video frames to the batch dimension to allow for batch arithm. operations.
		# Assumes input batch size = 1.
		x = x.view(-1, m)
		y, att_weights_ = self.att(x)

		y = y + x
		y = self.drop50(y)
		y = self.layer_norm_y(y)

		# Frame level importance score regression
		# Two layer NN
		y = self.ka(y)
		y = self.relu(y)
		y = self.drop50(y)
		y = self.layer_norm_ka(y)

		y = self.kd(y)
		y = self.sig(y)
		y = y.view(1, -1)

		return y, att_weights_



if __name__ == "__main__":
	pass

def weights_init(m):
	classname = m.__class__.__name__
	if classname == 'Linear':
		init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
		if m.bias is not None:
			init.constant_(m.bias, 0.1)



class HParameters:

    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.l2_req = 0.00001
        self.lr_epochs = [0]
        self.lr = [0.00005]

        self.epochs_max = 300
        self.train_batch_size = 1

        self.output_dir = 'ex-10'

        self.root = ''

        self.splits = ['splits/tvsum_splits.json',
                        'splits/summe_splits.json']

        self.splits += ['splits/tvsum_aug_splits.json',
                        'splits/summe_aug_splits.json']

        return


    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()

                setattr(self, key, val)
        self.datasets = self.datasets.split(',')

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)
class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out


    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split(dataset)
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File(dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename(splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None, f_len = 2048): #An EDIT 2021.06.01 pass feature length into vasnet model 
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.model = VASNet(m = f_len) #An EDIT 2021.06.01 pass feature length into vasnet model
        self.model.eval()
        self.model.apply(weights_init)
        #print(self.model)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0'):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]
        
        import time
        start = time.perf_counter()
        elapsed_epoch = -1

        lr = self.hps.lr[0]
        for epoch in range(self.hps.epochs_max):

            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                seq = dataset['features'][...]
                
                seq = torch.from_numpy(seq).unsqueeze(0)

                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                # print(seq.shape)
                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                loss_att = 0
                # print(i, key, target)

                # print(y.shape, target.shape)
                loss = criterion(y, target)

                # print(loss)
                # loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss.append([float(loss), float(loss_att)])

           
            # Evaluate test dataset
            val_fscore, video_scores = self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch

            avg_loss = np.array(avg_loss)

            if epoch % 20 == 0 : 
                print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='') #AN EDIT: only report for every 50 epoch
                print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')  #AN EDIT: only report for every 50 epoch
                print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))  #AN EDIT: only report for every 50 epoch
                print(f"{epoch - elapsed_epoch} epochs in {time.perf_counter() - start} seconds, {(time.perf_counter() - start)/(epoch - elapsed_epoch)} seconds per epoch")
                elapsed_epoch = epoch
                start = time.perf_counter()

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])

            # Save model weights
            path, filename = os.path.split(self.split_file)
            base_filename, _ = os.path.splitext(filename)
            path = os.path.join(output_dir, 'models_temp', base_filename+'_'+str(self.split_id))
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))

        return max_val_fscore, max_val_fscore_epoch


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        f_score, video_scores = self.eval_summary(summary, keys, metric=self.dataset_name,
                    results_filename=results_filename, att_vecs=att_vecs)

        return f_score, video_scores


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None):

        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]

            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores
