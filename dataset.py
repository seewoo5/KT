import os
import torch
from torch.utils.data import Dataset
import csv
from config import args
from util import create_full_path
from constant import *


class KTDataset(Dataset):

    def __init__(self, name, user_base_path, sample_infos, qid_to_embed_id, is_training):
        self._name = name # train, valid, test
        self._user_base_path = user_base_path
        self._sample_infos = sample_infos # list of [user_path, start_index]
        self._qid_to_embed_id = qid_to_embed_id # qid mapping dict
        self._is_training = is_training

    def get_sequence_info(self, start_index, max_index, seq_size):
        end_index = min(start_index + seq_size - 1, max_index)
        pad_counts = seq_size - (end_index - start_index + 1)
        return end_index, pad_counts

    def get_sequence(self, user_base_path, sample, qid_to_embed_id, is_training):
        user_path, start_index = sample
        user_full_path = create_full_path(user_base_path, user_path)

        with open(user_full_path, 'r') as f:
            data = f.readlines()
            user_data_length = len(data)

        end_index, pad_counts = self.get_sequence_info(start_index, user_data_length - 1, args.seq_size)
        paddings = [PAD_INDEX] * (pad_counts + 1) #

        input_list = []
        label_list = []
        is_predicted_list = []

        for idx, line in enumerate(data[start_index:end_index + 1], start=start_index):
            line = line.rstrip().split(',')
            question_id = int(line[1])
            embed_id = self._qid_to_embed_id[question_id]
            is_correct = int(line[2] == line[3])

            # label_list.append(is_correct)
            if idx == end_index:
                last_is_correct = is_correct
                target_id = embed_id
            else:
                if is_correct == 1:
                    input_list.append(embed_id)
                else:
                    input_list.append(embed_id + QUESTION_NUM)

        input_list += paddings
        assert len(input_list) == args.seq_size, "sequence size error"

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
                            self._qid_to_embed_id,
                            self._is_training)
