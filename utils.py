import logging
from os import makedirs
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import argparse
from os import path as osp

def setup_logger(output_dir:str):
    makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(osp.join(output_dir, 'debug.log'))])
    
def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class Examplear_Dataset(Dataset):
    def __init__(self, exemplar_set: list, train_transform, test_transform):
        '''
        [0]: images[index]
        [1]: saliency map
        [2]: psuedo label
        '''
        super(Dataset,self).__init__()
        self.exemplar_set = list()
        for i in exemplar_set:
            self.exemplar_set.extend(i)
        self.train_transform = train_transform
        self.test_transform = test_transform
    
    def __len__(self):
        return len(self.exemplar_set)

    def __getitem__(self, index):
        img, saliency_map, label = self.exemplar_set[index]
        return self.train_transform(Image.fromarray(img)), self.test_transform(Image.fromarray(img)), saliency_map, torch.tensor(label)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_task1', action='store_true')
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()