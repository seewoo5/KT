import os
import torch
from torch.utils.data import Dataset
import csv
from config import ARGS
from util import create_full_path
from constant import *


class KTDataset(Dataset):

    def __init__(self, name, user_base_path, sample_infos, qid_to_embed_id):
        self._name = name # train, valid, test
        self._user_base_path = user_base_path
        self._sample_infos = sample_infos # list of [user_path, target_index]
        self._qid_to_embed_id = qid_to_embed_id # qid mapping dict

    def get_sequence(self, user_base_path, sample, qid_to_embed_id):
        user_path, target_index = sample
        user_full_path = create_full_path(user_base_path, user_path)

        with open(user_full_path, 'r') as f:
            data = f.readlines() # no header
            data = data[:target_index+1]
            user_data_length = len(data)

        if user_data_length > ARGS.seq_size + 1:
            data = data[-(ARGS.seq_size + 1):]
            pad_counts = 0
        else:
            pad_counts = ARGS.seq_size + 1 - user_data_length

        input_list = []

        for idx, line in enumerate(data):
            line = line.rstrip().split(',')
            question_id = int(line[1])
            embed_id = self._qid_to_embed_id[question_id]
            is_correct = int(line[2] == line[3])

            if idx == len(data) - 1:
                last_is_correct = is_correct
                target_id = embed_id
            else:
                if is_correct == 1:
                    input_list.append(embed_id)
                else:
                    input_list.append(embed_id + QUESTION_NUM['modified_AAAI20'])

        paddings = [PAD_INDEX] * pad_counts
        input_list = paddings + input_list
        assert len(input_list) == ARGS.seq_size, "sequence size error"

        return {
            'label' : torch.Tensor([last_is_correct]).long(),
            'input': torch.Tensor(input_list).long(),
            'target_id': torch.Tensor([target_id - 1]).long()
        }

    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._user_base_path,
                            self._sample_infos[index],
                            self._qid_to_embed_id)
