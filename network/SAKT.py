import torch
import torch.nn as nn
from constant import PAD_INDEX
from network.transformer.Models import Transformer


class SAKT(Transformer):
    def __init__(self, question_num, input_dim, num_layers, num_head,
                 inner_dim, key_dim, value_dim, dropout, num_position):

        super().__init__(n_src_vocab=2*question_num+1, n_trg_vocab=question_num+1,
                         src_pad_idx=PAD_INDEX, trg_pad_idx=PAD_INDEX,
                         d_word_vec=input_dim, d_model=input_dim, d_inner=inner_dim,
                         n_layers=num_layers, n_head=num_head, d_k=key_dim, d_v=value_dim,
                         dropout=dropout, n_position=num_position)


    def _transform_interaction_to_question(self, interaction):
        pass

    def forward(self, input, target_id):
        pass