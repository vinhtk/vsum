## Paste all into one file so I can run in google colab easier
## Dependency: tqdm, tensorboardX, 
                                                                                                         
                                                                                                         
#                                                            ffffffffffffffff    iiii                      
#                                                           f::::::::::::::::f  i::::i                     
#                                                          f::::::::::::::::::f  iiii                      
#                                                          f::::::fffffff:::::f                            
#     cccccccccccccccc   ooooooooooo   nnnn  nnnnnnnn      f:::::f       ffffffiiiiiii    ggggggggg   ggggg
#   cc:::::::::::::::c oo:::::::::::oo n:::nn::::::::nn    f:::::f             i:::::i   g:::::::::ggg::::g
#  c:::::::::::::::::co:::::::::::::::on::::::::::::::nn  f:::::::ffffff        i::::i  g:::::::::::::::::g
# c:::::::cccccc:::::co:::::ooooo:::::onn:::::::::::::::n f::::::::::::f        i::::i g::::::ggggg::::::gg
# c::::::c     ccccccco::::o     o::::o  n:::::nnnn:::::n f::::::::::::f        i::::i g:::::g     g:::::g 
# c:::::c             o::::o     o::::o  n::::n    n::::n f:::::::ffffff        i::::i g:::::g     g:::::g 
# c:::::c             o::::o     o::::o  n::::n    n::::n  f:::::f              i::::i g:::::g     g:::::g 
# c::::::c     ccccccco::::o     o::::o  n::::n    n::::n  f:::::f              i::::i g::::::g    g:::::g 
# c:::::::cccccc:::::co:::::ooooo:::::o  n::::n    n::::n f:::::::f            i::::::ig:::::::ggggg:::::g 
#  c:::::::::::::::::co:::::::::::::::o  n::::n    n::::n f:::::::f            i::::::i g::::::::::::::::g 
#   cc:::::::::::::::c oo:::::::::::oo   n::::n    n::::n f:::::::f            i::::::i  gg::::::::::::::g 
#     cccccccccccccccc   ooooooooooo     nnnnnn    nnnnnn fffffffff            iiiiiiii    gggggggg::::::g 
#                                                                                                  g:::::g 
#                                                                                      gggggg      g:::::g 
#                                                                                      g:::::gg   gg:::::g 
#                                                                                       g::::::ggg:::::::g 
#                                                                                        gg:::::::::::::g  
#                                                                                          ggg::::::ggg    
#                                                                                             gggggg       


import argparse
from pathlib import Path
import sys
import pprint
import math

# save_dir = Path('ac-sum-gan-retrain/')

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.termination_point = math.floor(0.15*self.action_state_size)
        self.set_dataset_dir(self.video_type)
    
    def set_dataset_dir(self, video_type='TVSum'):
        dir = Path(self.output_dir)
        self.log_dir = dir.joinpath(video_type, 'logs/split' + str(self.split_index))
        self.score_dir = dir.joinpath(video_type, 'results/split' + str(self.split_index))
        self.save_dir = dir.joinpath(video_type, 'models/split' + str(self.split_index))
        for i in [self.log_dir, self.score_dir, self.save_dir]:
            i.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--data_file', type=str, default='train')
    parser.add_argument('--split_file', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='./ac-sum-gan-retrain/')
    parser.add_argument('--mode', type=str, default='train')


    parser.add_argument('--verbose', type=str2bool, default='true')
    parser.add_argument('--video_type', type=str, default='TVSum')

    # Model
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--regularization_factor', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.1)

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-5)
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--action_state_size', type=int, default=60)
    
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs) 


                                                                                                                                                                                                                     
#             dddddddd                                                                                                                                                                                                 
#             d::::::d  iiii                                                             iiii                                                                       tttt                                               
#             d::::::d i::::i                                                           i::::i                                                                   ttt:::t                                               
#             d::::::d  iiii                                                             iiii                                                                    t:::::t                                               
#             d:::::d                                                                                                                                            t:::::t                                               
#     ddddddddd:::::d iiiiiii     ssssssssss       ccccccccccccccccrrrrr   rrrrrrrrr   iiiiiii    mmmmmmm    mmmmmmm     aaaaaaaaaaaaa   nnnn  nnnnnnnn    ttttttt:::::ttttttt       ooooooooooo   rrrrr   rrrrrrrrr   
#   dd::::::::::::::d i:::::i   ss::::::::::s    cc:::::::::::::::cr::::rrr:::::::::r  i:::::i  mm:::::::m  m:::::::mm   a::::::::::::a  n:::nn::::::::nn  t:::::::::::::::::t     oo:::::::::::oo r::::rrr:::::::::r  
#  d::::::::::::::::d  i::::i ss:::::::::::::s  c:::::::::::::::::cr:::::::::::::::::r  i::::i m::::::::::mm::::::::::m  aaaaaaaaa:::::a n::::::::::::::nn t:::::::::::::::::t    o:::::::::::::::or:::::::::::::::::r 
# d:::::::ddddd:::::d  i::::i s::::::ssss:::::sc:::::::cccccc:::::crr::::::rrrrr::::::r i::::i m::::::::::::::::::::::m           a::::a nn:::::::::::::::ntttttt:::::::tttttt    o:::::ooooo:::::orr::::::rrrrr::::::r
# d::::::d    d:::::d  i::::i  s:::::s  ssssss c::::::c     ccccccc r:::::r     r:::::r i::::i m:::::mmm::::::mmm:::::m    aaaaaaa:::::a   n:::::nnnn:::::n      t:::::t          o::::o     o::::o r:::::r     r:::::r
# d:::::d     d:::::d  i::::i    s::::::s      c:::::c              r:::::r     rrrrrrr i::::i m::::m   m::::m   m::::m  aa::::::::::::a   n::::n    n::::n      t:::::t          o::::o     o::::o r:::::r     rrrrrrr
# d:::::d     d:::::d  i::::i       s::::::s   c:::::c              r:::::r             i::::i m::::m   m::::m   m::::m a::::aaaa::::::a   n::::n    n::::n      t:::::t          o::::o     o::::o r:::::r            
# d:::::d     d:::::d  i::::i ssssss   s:::::s c::::::c     ccccccc r:::::r             i::::i m::::m   m::::m   m::::ma::::a    a:::::a   n::::n    n::::n      t:::::t    tttttto::::o     o::::o r:::::r            
# d::::::ddddd::::::ddi::::::is:::::ssss::::::sc:::::::cccccc:::::c r:::::r            i::::::im::::m   m::::m   m::::ma::::a    a:::::a   n::::n    n::::n      t::::::tttt:::::to:::::ooooo:::::o r:::::r            
#  d:::::::::::::::::di::::::is::::::::::::::s  c:::::::::::::::::c r:::::r            i::::::im::::m   m::::m   m::::ma:::::aaaa::::::a   n::::n    n::::n      tt::::::::::::::to:::::::::::::::o r:::::r            
#   d:::::::::ddd::::di::::::i s:::::::::::ss    cc:::::::::::::::c r:::::r            i::::::im::::m   m::::m   m::::m a::::::::::aa:::a  n::::n    n::::n        tt:::::::::::tt oo:::::::::::oo  r:::::r            
#    ddddddddd   dddddiiiiiiii  sssssssssss        cccccccccccccccc rrrrrrr            iiiiiiiimmmmmm   mmmmmm   mmmmmm  aaaaaaaaaa  aaaa  nnnnnn    nnnnnn          ttttttttttt     ooooooooooo    rrrrrrr            
                                                                                                                                                                                                                     
                                                                                                                                                                                                                     
# -*- coding: utf-8 -*-
import torch.nn as nn


class cLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, input_size]
        Return:
            last_h: [1, hidden_size]
        """
        self.lstm.flatten_parameters()

        # output: seq_len, batch, hidden_size * num_directions
        # h_n, c_n: num_layers * num_directions, batch_size, hidden_size
        output, (h_n, c_n) = self.lstm(features, init_hidden)

        # [batch_size, hidden_size]
        last_h = h_n[-1]

        return last_h


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator: cLSTM + output projection to probability"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h : [1, hidden_size]
                Last h from top layer of discriminator
            prob: [batch_size, 1]
                Probability to be original feature from CNN
        """

        # [1, hidden_size]
        h = self.cLSTM(features)

        prob = self.out(h).squeeze()

        return h, prob
                                                                                                                                  
                                                                                                                                                                                          
#     ssssssssss   uuuuuu    uuuuuu     mmmmmmm    mmmmmmm      mmmmmmm    mmmmmmm     aaaaaaaaaaaaa   rrrrr   rrrrrrrrr   iiiiiii zzzzzzzzzzzzzzzzz    eeeeeeeeeeee    rrrrr   rrrrrrrrr   
#   ss::::::::::s  u::::u    u::::u   mm:::::::m  m:::::::mm  mm:::::::m  m:::::::mm   a::::::::::::a  r::::rrr:::::::::r  i:::::i z:::::::::::::::z  ee::::::::::::ee  r::::rrr:::::::::r  
# ss:::::::::::::s u::::u    u::::u  m::::::::::mm::::::::::mm::::::::::mm::::::::::m  aaaaaaaaa:::::a r:::::::::::::::::r  i::::i z::::::::::::::z  e::::::eeeee:::::eer:::::::::::::::::r 
# s::::::ssss:::::su::::u    u::::u  m::::::::::::::::::::::mm::::::::::::::::::::::m           a::::a rr::::::rrrrr::::::r i::::i zzzzzzzz::::::z  e::::::e     e:::::err::::::rrrrr::::::r
#  s:::::s  ssssss u::::u    u::::u  m:::::mmm::::::mmm:::::mm:::::mmm::::::mmm:::::m    aaaaaaa:::::a  r:::::r     r:::::r i::::i       z::::::z   e:::::::eeeee::::::e r:::::r     r:::::r
#    s::::::s      u::::u    u::::u  m::::m   m::::m   m::::mm::::m   m::::m   m::::m  aa::::::::::::a  r:::::r     rrrrrrr i::::i      z::::::z    e:::::::::::::::::e  r:::::r     rrrrrrr
#       s::::::s   u::::u    u::::u  m::::m   m::::m   m::::mm::::m   m::::m   m::::m a::::aaaa::::::a  r:::::r             i::::i     z::::::z     e::::::eeeeeeeeeee   r:::::r            
# ssssss   s:::::s u:::::uuuu:::::u  m::::m   m::::m   m::::mm::::m   m::::m   m::::ma::::a    a:::::a  r:::::r             i::::i    z::::::z      e:::::::e            r:::::r            
# s:::::ssss::::::su:::::::::::::::uum::::m   m::::m   m::::mm::::m   m::::m   m::::ma::::a    a:::::a  r:::::r            i::::::i  z::::::zzzzzzzze::::::::e           r:::::r            
# s::::::::::::::s  u:::::::::::::::um::::m   m::::m   m::::mm::::m   m::::m   m::::ma:::::aaaa::::::a  r:::::r            i::::::i z::::::::::::::z e::::::::eeeeeeee   r:::::r            
#  s:::::::::::ss    uu::::::::uu:::um::::m   m::::m   m::::mm::::m   m::::m   m::::m a::::::::::aa:::a r:::::r            i::::::iz:::::::::::::::z  ee:::::::::::::e   r:::::r            
#   sssssssssss        uuuuuuuu  uuuummmmmm   mmmmmm   mmmmmmmmmmmm   mmmmmm   mmmmmm  aaaaaaaaaa  aaaa rrrrrrr            iiiiiiiizzzzzzzzzzzzzzzzz    eeeeeeeeeeeeee   rrrrrrr            
                                                                                                                                                                                          
                                                                                                                                                                                          
                                                                                                                                                                                          
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

# from layers.lstmcell import StackedLSTMCell
class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, hidden_size] (compressed pool5 features)
        Return:
            scores: [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            last hidden:
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len: scalar (int)
            init_hidden:
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda()

        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h: [2=num_layers, 1, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        """
        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        # [num_layers, hidden_size]
        h = h.squeeze(1)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features):
        """
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            h_mu: [num_layers=2, hidden_size]
            h_log_variance: [num_layers=2, hidden_size]
            decoded_features: [seq_len, 1, hidden_size]
        """
        
        # Apply weights
        # [seq_len, 1]
        scores = self.s_lstm(image_features)

        # [seq_len, 1, hidden_size]
        weighted_features = image_features * scores.view(-1, 1, 1)

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features

                                                                                                                                                            
                                                                                                                                                                                          
                                                                                                                                                                                          
                                                                                                                                                                                          
                                                                                                                                                                  
                                                                                                                                                                  
#    SSSSSSSSSSSSSSS                  lllllll                                                                   
#  SS:::::::::::::::S                 l:::::l                                                                   
# S:::::SSSSSS::::::S                 l:::::l                                                                   
# S:::::S     SSSSSSS                 l:::::l                                                                   
# S:::::S               ooooooooooo    l::::l vvvvvvv           vvvvvvv    eeeeeeeeeeee    rrrrr   rrrrrrrrr    
# S:::::S             oo:::::::::::oo  l::::l  v:::::v         v:::::v   ee::::::::::::ee  r::::rrr:::::::::r   
#  S::::SSSS         o:::::::::::::::o l::::l   v:::::v       v:::::v   e::::::eeeee:::::eer:::::::::::::::::r  
#   SS::::::SSSSS    o:::::ooooo:::::o l::::l    v:::::v     v:::::v   e::::::e     e:::::err::::::rrrrr::::::r 
#     SSS::::::::SS  o::::o     o::::o l::::l     v:::::v   v:::::v    e:::::::eeeee::::::e r:::::r     r:::::r 
#        SSSSSS::::S o::::o     o::::o l::::l      v:::::v v:::::v     e:::::::::::::::::e  r:::::r     rrrrrrr 
#             S:::::So::::o     o::::o l::::l       v:::::v:::::v      e::::::eeeeeeeeeee   r:::::r             
#             S:::::So::::o     o::::o l::::l        v:::::::::v       e:::::::e            r:::::r             
# SSSSSSS     S:::::So:::::ooooo:::::ol::::::l        v:::::::v        e::::::::e           r:::::r             
# S::::::SSSSSS:::::So:::::::::::::::ol::::::l         v:::::v          e::::::::eeeeeeee   r:::::r             
# S:::::::::::::::SS  oo:::::::::::oo l::::::l          v:::v            ee:::::::::::::e   r:::::r             
#  SSSSSSSSSSSSSSS      ooooooooooo   llllllll           vvv               eeeeeeeeeeeeee   rrrrrrr             
                                                                                                                                            
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm, trange
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# from layers import Summarizer, Discriminator
# from utils import TensorboardWriter
from tensorboardX import SummaryWriter
class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        """
        Extended SummaryWriter Class from tensorboard-pytorch (tensorbaordX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer
        """
        super(TensorboardWriter, self).__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module, step_i):
        """
        module: nn.Module
        """
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        self.add_histogram(name, values, step_i)

# from layers.actor_critic import Actor, Critic
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        """Actor that picks a fragment for the summary in every iteration"""
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, self.action_size)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            distribution: categorical distribution of pytorch
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = self.linear4(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        """Critic that evaluates the Actor's choices"""
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Args:
            state: [num_fragments, 1]
        Return:
            value: scalar
        """
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.relu(self.linear3(output))
        output = F.relu(self.linear4(output))
        value = self.linear5(output)
        return value


# from fragments import calculate_fragments

# labels for training the GAN part of the model
original_label = torch.tensor(1.0).cuda()
summary_label = torch.tensor(0.0).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_returns(next_value, rewards, masks, gamma=0.99):
    """ Function that computes the return z_i following the equation (6) of the paper"""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates AC-SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.actor = Actor(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size).cuda()
        self.critic = Critic(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator, self.actor, self.critic])

        if self.config.mode == 'train':
            # Build Optimizers
            self.e_optimizer = optim.Adam(
                self.summarizer.vae.e_lstm.parameters(),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                self.summarizer.vae.d_lstm.parameters(),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)
            self.optimizerA_s = optim.Adam(list(self.actor.parameters())
                                           + list(self.summarizer.s_lstm.parameters())
                                           + list(self.linear_compress.parameters()),
                                           lr=self.config.lr)
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.config.lr)

            self.writer = TensorboardWriter(str(self.config.log_dir))

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_sum, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    criterion = nn.MSELoss()

    def AC(self, original_features, seq_len, action_fragments):
        """ Function that makes the actor's actions, in the training steps where the actor and critic components are not trained"""
        scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

        fragment_scores = np.zeros(self.config.action_state_size)  # [num_fragments, 1]
        for fragment in range(self.config.action_state_size):
            fragment_scores[fragment] = scores[action_fragments[fragment,0]:action_fragments[fragment,1]+1].mean()
        state = fragment_scores

        previous_actions = []  # save all the actions (the selected fragments of each episode)
        reduction_factor = (self.config.action_state_size - self.config.termination_point) / self.config.action_state_size
        action_scores = (torch.ones(seq_len) * reduction_factor).cuda()
        action_fragment_scores = (torch.ones(self.config.action_state_size)).cuda()

        counter = 0
        for ACstep in range(self.config.termination_point):

            state = torch.FloatTensor(state).cuda()
            # select an action
            dist = self.actor(state)
            action = dist.sample()  # returns a scalar between 0-action_state_size

            if action not in previous_actions:
                previous_actions.append(action)
                action_factor = (self.config.termination_point - counter) / (self.config.action_state_size - counter) + 1

                action_scores[action_fragments[action, 0]:action_fragments[action, 1] + 1] = action_factor
                action_fragment_scores[action] = 0

                counter = counter + 1

            next_state = state * action_fragment_scores
            next_state = next_state.cpu().detach().numpy()
            state = next_state

        weighted_scores = action_scores.unsqueeze(1) * scores
        weighted_features = weighted_scores.view(-1, 1, 1) * original_features

        return weighted_features, weighted_scores

    def train(self):

        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            recon_loss_init_history = []
            recon_loss_history = []
            sparsity_loss_history = []
            prior_loss_history = []
            g_loss_history = []
            e_loss_history = []
            d_loss_history = []
            c_original_loss_history = []
            c_summary_loss_history = []
            actor_loss_history = []
            critic_loss_history = []
            reward_history = []            
            
            # Train in batches of as many videos as the batch_size
            num_batches = int(len(self.train_loader)/self.config.batch_size)
            iterator = iter(self.train_loader)
            for batch in range(num_batches):
                list_image_features = []
                list_action_fragments = []
                
                print(f'batch: {batch}')
                
                # ---- Train eLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training eLSTM...')
                self.e_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features, action_fragments = next(iterator)
                    
                    action_fragments = action_fragments.squeeze(0)
                    # [batch_size, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    
                    list_image_features.append(image_features)
                    list_action_fragments.append(action_fragments)
    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
    
                    weighted_features, scores = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    if self.config.verbose:
                        tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    prior_loss = self.prior_loss(h_mu, h_log_variance)
    
                    tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}')
    
                    e_loss = reconstruction_loss + prior_loss
                    e_loss = e_loss/self.config.batch_size
                    e_loss.backward()
                    
                    prior_loss_history.append(prior_loss.data)
                    e_loss_history.append(e_loss.data)
                    
                # Update e_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.e_lstm.parameters(), self.config.clip)
                self.e_optimizer.step()
                
                #---- Train dLSTM (decoder/generator) ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')
                self.d_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
    
                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)
    
                    tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    g_loss = self.criterion(sum_prob, original_label)
                    
                    orig_features = original_features.squeeze(1)    # [seq_len, hidden_size]
                    gen_features = generated_features.squeeze(1)    #         >>
                    recon_losses = []
                    for frame_index in range(seq_len):
                        recon_losses.append(self.reconstruction_loss(orig_features[frame_index,:], gen_features[frame_index,:]))
                    reconstruction_loss_init = torch.stack(recon_losses).mean()

                    if self.config.verbose:
                        tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, g loss: {g_loss.item():.3f}')
                    
                    d_loss = reconstruction_loss + g_loss
                    d_loss = d_loss/self.config.batch_size
                    d_loss.backward()
                    
                    recon_loss_init_history.append(reconstruction_loss_init.data)
                    recon_loss_history.append(reconstruction_loss.data)
                    g_loss_history.append(g_loss.data)
                    d_loss_history.append(d_loss.data)
                    
                # Update d_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.d_lstm.parameters(), self.config.clip)
                self.d_optimizer.step()
                
                #---- Train cLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training cLSTM...')
                self.c_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # Train with original loss
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    h_origin, original_prob = self.discriminator(original_features)
                    c_original_loss = self.criterion(original_prob, original_label)
                    c_original_loss = c_original_loss/self.config.batch_size
                    c_original_loss.backward()
    
                    # Train with summary loss
                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
                    h_sum, sum_prob = self.discriminator(generated_features.detach())
                    c_summary_loss = self.criterion(sum_prob, summary_label)
                    c_summary_loss = c_summary_loss/self.config.batch_size
                    c_summary_loss.backward()
                    
                    tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
                    
                    c_original_loss_history.append(c_original_loss.data)
                    c_summary_loss_history.append(c_summary_loss.data)
                    
                # Update c_lstm parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(list(self.discriminator.parameters()) + list(self.linear_compress.parameters()), self.config.clip)
                self.c_optimizer.step()
                
                #---- Train sLSTM and actor-critic ----#
                if self.config.verbose:
                    tqdm.write('Training sLSTM, actor and critic...')
                self.optimizerA_s.zero_grad()
                self.optimizerC.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]
                    
                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).cuda()
                    seq_len = image_features_.shape[0]
                    
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

                    fragment_scores = np.zeros(self.config.action_state_size)  # [num_fragments, 1]
                    for fragment in range(self.config.action_state_size):
                        fragment_scores[fragment] = scores[action_fragments[fragment, 0]:action_fragments[fragment, 1] + 1].mean()
    
                    state = fragment_scores  # [action_state_size, 1]
    
                    previous_actions = []  # save all the actions (the selected fragments of each step)
                    reduction_factor = (self.config.action_state_size - self.config.termination_point) / self.config.action_state_size
                    action_scores = (torch.ones(seq_len) * reduction_factor).cuda()
                    action_fragment_scores = (torch.ones(self.config.action_state_size)).cuda()
    
                    log_probs = []
                    values = []
                    rewards = []
                    masks = []
                    entropy = 0
    
                    counter = 0
                    for ACstep in range(self.config.termination_point):
                        # select an action, get a value for the current state
                        state = torch.FloatTensor(state).cuda()  # [action_state_size, 1]
                        dist, value = self.actor(state), self.critic(state)
                        action = dist.sample()  # returns a scalar between 0-action_state_size
    
                        if action in previous_actions:
    
                            reward = 0
    
                        else:
    
                            previous_actions.append(action)
                            action_factor = (self.config.termination_point - counter) / (self.config.action_state_size - counter) + 1
    
                            action_scores[action_fragments[action, 0]:action_fragments[action, 1] + 1] = action_factor
                            action_fragment_scores[action] = 0
    
                            weighted_scores = action_scores.unsqueeze(1) * scores
                            weighted_features = weighted_scores.view(-1, 1, 1) * original_features
    
                            h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
    
                            h_origin, original_prob = self.discriminator(original_features)
                            h_sum, sum_prob = self.discriminator(generated_features)
    
                            tqdm.write(f'original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}')
    
                            rec_loss = self.reconstruction_loss(h_origin, h_sum)
                            reward = 1 - rec_loss.item()  # the less the distance, the higher the reward
                            counter = counter + 1
    
                        next_state = state * action_fragment_scores
                        next_state = next_state.cpu().detach().numpy()
    
                        log_prob = dist.log_prob(action).unsqueeze(0)
                        entropy += dist.entropy().mean()
    
                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
    
                        if ACstep == self.config.termination_point-1 :
                            masks.append(torch.tensor([0], dtype=torch.float, device=device)) 
                        else:
                            masks.append(torch.tensor([1], dtype=torch.float, device=device))
    
                        state = next_state
    
                    next_state = torch.FloatTensor(next_state).to(device)
                    next_value = self.critic(next_state)
                    returns = compute_returns(next_value, rewards, masks)
    
                    log_probs = torch.cat(log_probs)
                    returns = torch.cat(returns).detach()
                    values = torch.cat(values)
    
                    advantage = returns - values
    
                    actor_loss = -((log_probs * advantage.detach()).mean() + (self.config.entropy_coef/self.config.termination_point)*entropy)
                    sparsity_loss = self.sparsity_loss(scores)
                    critic_loss = advantage.pow(2).mean()
                    
                    actor_loss = actor_loss/self.config.batch_size
                    sparsity_loss = sparsity_loss/self.config.batch_size
                    critic_loss = critic_loss/self.config.batch_size
                    actor_loss.backward()
                    sparsity_loss.backward()
                    critic_loss.backward()
                    
                    reward_mean = torch.mean(torch.stack(rewards))
                    reward_history.append(reward_mean)
                    actor_loss_history.append(actor_loss)
                    sparsity_loss_history.append(sparsity_loss)
                    critic_loss_history.append(critic_loss)
                    
                    if self.config.verbose:
                        tqdm.write('Plotting...')
    
                    self.writer.update_loss(original_prob.data, step, 'original_prob')
                    self.writer.update_loss(sum_prob.data, step, 'sum_prob')
    
                    step += 1
                    
                # Update s_lstm, actor and critic parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.linear_compress.parameters())
                                           + list(self.summarizer.s_lstm.parameters())+list(self.critic.parameters()), self.config.clip)
                self.optimizerA_s.step()
                self.optimizerC.step()
                

            recon_loss_init = torch.stack(recon_loss_init_history).mean()
            recon_loss = torch.stack(recon_loss_history).mean()
            prior_loss = torch.stack(prior_loss_history).mean()
            g_loss = torch.stack(g_loss_history).mean()
            e_loss = torch.stack(e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_original_loss = torch.stack(c_original_loss_history).mean()
            c_summary_loss = torch.stack(c_summary_loss_history).mean()
            sparsity_loss = torch.stack(sparsity_loss_history).mean()
            actor_loss = torch.stack(actor_loss_history).mean()
            critic_loss = torch.stack(critic_loss_history).mean()
            reward = torch.mean(torch.stack(reward_history))

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(recon_loss_init, epoch_i, 'recon_loss_init_epoch')
            self.writer.update_loss(recon_loss, epoch_i, 'recon_loss_epoch')
            self.writer.update_loss(prior_loss, epoch_i, 'prior_loss_epoch')    
            self.writer.update_loss(g_loss, epoch_i, 'g_loss_epoch')    
            self.writer.update_loss(e_loss, epoch_i, 'e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_original_loss, epoch_i, 'c_original_loss_epoch')
            self.writer.update_loss(c_summary_loss, epoch_i, 'c_summary_loss_epoch')
            self.writer.update_loss(sparsity_loss, epoch_i, 'sparsity_loss_epoch')
            self.writer.update_loss(actor_loss, epoch_i, 'actor_loss_epoch')
            self.writer.update_loss(critic_loss, epoch_i, 'critic_loss_epoch')
            self.writer.update_loss(reward, epoch_i, 'reward_epoch')

            # Save parameters at checkpoint
            ckpt_path = self.config.save_dir.joinpath(f'/epoch-{epoch_i}.pkl')
            ckpt_path.touch(exist_ok=True)
            if self.config.verbose:
                tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)

    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for image_features, video_name, action_fragments in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, batch_size=1, input_size)]
            image_features = image_features.view(-1, self.config.input_size)
            image_features_ = Variable(image_features).cuda()

            # [seq_len, 1, hidden_size]
            original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
            seq_len = original_features.shape[0]
            
            with torch.no_grad():
                # print("Try to find the warningp ", file=sys.stderr)
                _, scores = self.AC(original_features, seq_len, action_fragments)

                scores = scores.squeeze(1)
                scores = scores.cpu().numpy().tolist()

                out_dict[video_name] = scores

            self.config.score_dir.mkdir(parents=True, exist_ok= True)
            score_save_path = self.config.score_dir.joinpath(
                f'{self.config.video_type}_{epoch_i}.json')
            score_save_path.touch(mode=0x777, exist_ok=True)
            # if not os.path.exists(score_save_path): os.makedirs(score_save_path)
            with open(score_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            
                                                                                                                                                                                                              
#             dddddddd                                                                                                                   dddddddd                                        
#             d::::::d                           tttt                             lllllll                                                d::::::d                                        
#             d::::::d                        ttt:::t                             l:::::l                                                d::::::d                                        
#             d::::::d                        t:::::t                             l:::::l                                                d::::::d                                        
#             d:::::d                         t:::::t                             l:::::l                                                d:::::d                                         
#     ddddddddd:::::d   aaaaaaaaaaaaa   ttttttt:::::ttttttt      aaaaaaaaaaaaa     l::::l    ooooooooooo     aaaaaaaaaaaaa       ddddddddd:::::d     eeeeeeeeeeee    rrrrr   rrrrrrrrr   
#   dd::::::::::::::d   a::::::::::::a  t:::::::::::::::::t      a::::::::::::a    l::::l  oo:::::::::::oo   a::::::::::::a    dd::::::::::::::d   ee::::::::::::ee  r::::rrr:::::::::r  
#  d::::::::::::::::d   aaaaaaaaa:::::a t:::::::::::::::::t      aaaaaaaaa:::::a   l::::l o:::::::::::::::o  aaaaaaaaa:::::a  d::::::::::::::::d  e::::::eeeee:::::eer:::::::::::::::::r 
# d:::::::ddddd:::::d            a::::a tttttt:::::::tttttt               a::::a   l::::l o:::::ooooo:::::o           a::::a d:::::::ddddd:::::d e::::::e     e:::::err::::::rrrrr::::::r
# d::::::d    d:::::d     aaaaaaa:::::a       t:::::t              aaaaaaa:::::a   l::::l o::::o     o::::o    aaaaaaa:::::a d::::::d    d:::::d e:::::::eeeee::::::e r:::::r     r:::::r
# d:::::d     d:::::d   aa::::::::::::a       t:::::t            aa::::::::::::a   l::::l o::::o     o::::o  aa::::::::::::a d:::::d     d:::::d e:::::::::::::::::e  r:::::r     rrrrrrr
# d:::::d     d:::::d  a::::aaaa::::::a       t:::::t           a::::aaaa::::::a   l::::l o::::o     o::::o a::::aaaa::::::a d:::::d     d:::::d e::::::eeeeeeeeeee   r:::::r            
# d:::::d     d:::::d a::::a    a:::::a       t:::::t    tttttta::::a    a:::::a   l::::l o::::o     o::::oa::::a    a:::::a d:::::d     d:::::d e:::::::e            r:::::r            
# d::::::ddddd::::::dda::::a    a:::::a       t::::::tttt:::::ta::::a    a:::::a  l::::::lo:::::ooooo:::::oa::::a    a:::::a d::::::ddddd::::::dde::::::::e           r:::::r            
#  d:::::::::::::::::da:::::aaaa::::::a       tt::::::::::::::ta:::::aaaa::::::a  l::::::lo:::::::::::::::oa:::::aaaa::::::a  d:::::::::::::::::d e::::::::eeeeeeee   r:::::r            
#   d:::::::::ddd::::d a::::::::::aa:::a        tt:::::::::::tt a::::::::::aa:::a l::::::l oo:::::::::::oo  a::::::::::aa:::a  d:::::::::ddd::::d  ee:::::::::::::e   r:::::r            
#    ddddddddd   ddddd  aaaaaaaaaa  aaaa          ttttttttttt    aaaaaaaaaa  aaaa llllllll   ooooooooooo     aaaaaaaaaa  aaaa   ddddddddd   ddddd    eeeeeeeeeeeeee   rrrrrrr            
                                                                                                                                                                                                           
                                                                                                                                                                                                              
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

def calculate_fragments(sequence_len, num_fragments):
    
    '''
    The sequence must be divided into "num_fragments" fragments.
    Since seq_len/num won't be a perfect division, we take both
    floor and ceiling parts, in a way such that the sum of all
    fragments will be equal to the total sequence.'''
    
    fragment_size = sequence_len/num_fragments
    fragment_floor = math.floor(fragment_size)
    fragment_ceil = math.ceil(fragment_size)
    i_part, d_part = divmod(fragment_size, 1)
    
    frag_jump = np.zeros(num_fragments)

    upper = d_part * num_fragments
    upper = np.round(upper).astype(int)
    lower = (1-d_part) * num_fragments
    lower = np.round(lower).astype(int)

    for i in range(lower):
        frag_jump[i] = fragment_floor
    for i in range(upper):
        frag_jump[lower+i] = fragment_ceil

    # Roll the scores, so that the larger fragments fall at 
    # the center of the sequence. Should not make a difference.
    frag_jump = np.roll(frag_jump, -int(num_fragments*(1-d_part)/2))

    if frag_jump[num_fragments-1] == 1:
        frag_jump[int(num_fragments/2)] = 1

    return frag_jump.astype(int)


def compute_fragments(seq_len, action_state_size):
    
    # "action_fragments" contains the starting and ending frame of each action fragment
    frag_jump = calculate_fragments(seq_len, action_state_size)
    action_fragments = torch.zeros((action_state_size,2), dtype=torch.int64)
    for i in range(action_state_size-1):
        action_fragments[i,1] = torch.tensor(sum(frag_jump[0:i+1])-1)
        action_fragments[i+1,0] = torch.tensor(sum(frag_jump[0:i+1]))
    action_fragments[action_state_size-1, 1] = torch.tensor(sum(frag_jump[0:action_state_size])-1)    
                
    return action_fragments

class VideoData(Dataset):
    def __init__(self, config):
    # def __init__(self, mode, split_index, action_state_size):
        self.mode = config.mode
        # self.name = 'tvsum'
        # self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
        #                  '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        # self.splits_filename = ['../data/splits/' + self.name + '_splits.json']


        
        self.split_index = config.split_index # it represents the current split (varies from 0 to 4)

        # if 'summe' in self.splits_filename[0]:
        #     self.filename = self.datasets[0]
        # elif 'tvsum' in self.splits_filename[0]:
        #     self.filename = self.datasets[1]
        hdf = h5py.File(config.data_file, 'r')
        self.action_fragments = {}
        self.list_features = []

        with open(config.split_file) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i==self.split_index:
                    self.split = split
                    
        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_features.append(features)
            self.action_fragments[video_name] = compute_fragments(features.shape[0], config.action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features and the action_fragments; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.split[self.mode + '_keys'][index]  #gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

def get_loader(config):
    if config.mode.lower() == 'train':
        vd = VideoData(config)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(config)
                                                   
                                                                    
#                                             iiii                    
#                                            i::::i                   
#                                             iiii                    
                                                                    
#    mmmmmmm    mmmmmmm     aaaaaaaaaaaaa   iiiiiii nnnn  nnnnnnnn    
#  mm:::::::m  m:::::::mm   a::::::::::::a  i:::::i n:::nn::::::::nn  
# m::::::::::mm::::::::::m  aaaaaaaaa:::::a  i::::i n::::::::::::::nn 
# m::::::::::::::::::::::m           a::::a  i::::i nn:::::::::::::::n
# m:::::mmm::::::mmm:::::m    aaaaaaa:::::a  i::::i   n:::::nnnn:::::n
# m::::m   m::::m   m::::m  aa::::::::::::a  i::::i   n::::n    n::::n
# m::::m   m::::m   m::::m a::::aaaa::::::a  i::::i   n::::n    n::::n
# m::::m   m::::m   m::::ma::::a    a:::::a  i::::i   n::::n    n::::n
# m::::m   m::::m   m::::ma::::a    a:::::a i::::::i  n::::n    n::::n
# m::::m   m::::m   m::::ma:::::aaaa::::::a i::::::i  n::::n    n::::n
# m::::m   m::::m   m::::m a::::::::::aa:::ai::::::i  n::::n    n::::n
# mmmmmm   mmmmmm   mmmmmm  aaaaaaaaaa  aaaaiiiiiiii  nnnnnn    nnnnnn
                                                                    

import json                                                        
import os
if __name__ == '__main__':
    config = get_config(mode='train'
        , data_file='for.training/datasets/eccv16_dataset_tvsum_google_pool5.h5'
        , split_file = 'for.training/datasets/tvsum_canonical_splits.json'
        , verbose=False
    )
    test_config = get_config(
        mode='test'
        , data_file='for.training/datasets/eccv16_dataset_tvsum_google_pool5.h5'
        , split_file = 'for.training/datasets/tvsum_canonical_splits.json'
        , verbose=False
    )
    
    print(config)
    print(test_config)

    if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)

    train_loader = get_loader(config)
    test_loader = get_loader(test_config)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.evaluate(-1)  # evaluates the summaries generated using the initial random weights of the network
    solver.train()
                              
                                                                    
                                                                    
                                                                    

