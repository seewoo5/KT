import os
import torch
from torch.utils.data import Dataset
import csv
from config import args
from util import create_full_path
from constant import *

class TripleLineDataset(Dataset):

    def __init__(self, name, sample_infos, dataset_name='ASSISTments2009'):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # list of (qid_list, is_correct_list)
        self._dataset_name = dataset_name

    def get_sequence(self, sample):
        # sample is a pair of qid_list & is_correct_list
        qid_list, is_correct_list = sample
        last_is_correct = is_correct_list[-1]
        target_id = qid_list[-1]

        input_list = qid_list[:-1]
        for i in range(len(input_list)):
            if is_correct_list[i] == 0:
                input_list[i] += question_num[self._dataset_name]

        if len(input_list) > args.seq_size:
            input_list = input_list[-args.seq_size:]
        else:
            pad_counts = args.seq_size - len(input_list)
            input_list = [PAD_INDEX] * pad_counts + input_list
        assert len(input_list) == args.seq_size, "sequence size error"

        return {
            'label': torch.Tensor([last_is_correct]).long(),
            'input': torch.Tensor(input_list).long(),
            'target_id': torch.Tensor([target_id - 1]).long()
        }

    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])
