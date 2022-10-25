"""
lukun199@gmail.com
19th Feb., 2021

# data_loader.py
"""

import os, pickle, torch
from torch.utils.data import Dataset
import numpy as np


class Dataset_sentence(Dataset):
    def __init__(self, _path, use_sos=False):

        if not _path: _path = r'H:\Europarl'  # change to your own path
        self._path = os.path.join(_path, 'english_vocab.pkl')
        self.dict = {}
        tmp = pickle.load(open(self._path, 'rb'))
        for kk,vv in tmp['voc'].items(): self.dict[kk] = vv+3
        # add sos, eos, and pad.
        self.dict['PAD'], self.dict['SOS'], self.dict['EOS'] = 0, 1, 2
        self.len_range = tmp['len_range']
        self.rev_dict = {vv: kk for kk, vv in self.dict.items()}
        sos_head = [1] if use_sos else []
        self.data_num = torch.tensor([sos_head + list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5!=0])  # use tmp['sent_str'][:1000] for debugging
        self.test_data_num = [sos_head + list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5==0]  # 20% of data
        self.data_len = np.array(list(map(lambda s: sum(s != 0), self.data_num)))

        with open('train_dict.pkl','wb') as f: pickle.dump(self.dict, f)
        print('[*]------------vocabulary size is:----', self.get_dict_len())
        print('[*]------------sentences size is:----', self.__len__())
        #print('[*]------------test sentences size is:----', len(self.test_data_num))

    def __getitem__(self, index):
        return self.data_num[index], self.data_len[index]

    def __len__(self):
        return len(self.data_num)

    def get_dict_len(self):
        return len(self.dict)


class Dataset_sentence_test(Dataset):
    def __init__(self, _path):
        if not _path: _path = r'H:\Europarl'
        self._path = os.path.join(_path, 'english_vocab.pkl')
        self.dict = {}
        tmp = pickle.load(open(self._path, 'rb'))
        for kk,vv in tmp['voc'].items(): self.dict[kk] = vv+3
        # add sos, eos, and pad.
        self.dict['PAD'], self.dict['SOS'], self.dict['EOS'] = 0, 1, 2
        self.len_range = tmp['len_range']
        self.rev_dict = {vv: kk for kk, vv in self.dict.items()}
        self.data_num = [[1] + list(map(lambda t:self.dict[t], x.split(' '))) + [2]
                         + (self.len_range[1]-len(x.split(' ')))*[0]
                         for idx, x in enumerate(tmp['sent_str']) if idx%5==0]
        print('[*]------------vocabulary size is:----', self.get_dict_len())
        print('[*]------------sentences size is:----', len(self.data_num))

    def __getitem__(self, index):
        return torch.tensor(self.data_num[index])

    def __len__(self):
        return len(self.data_num)

    def get_dict_len(self):
        return len(self.dict)


def collate_func(in_data):
    batch_tensor, batch_len = list(zip(*(sorted(in_data, key=lambda s:-s[1]))))
    return torch.stack(batch_tensor, dim=0), batch_len

