# TODO
import torch
import torch.nn as nn
from constant import PAD_INDEX


class DKVMN(nn.Module):
    def __init__(self, key_dim, value_dim, summary_dim, question_num, concept_num):
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._summary_dim = summary_dim
        self._question_num = question_num
        self._concept_num = concept_num

        self._question_embedding = nn.Embedding(num_embeddings=question_num+1,
                                                embedding_dim=key_dim,
                                                padding_idx=PAD_INDEX)
        self._interaction_embedding = nn.Embedding(num_embeddings=2*question_num+1,
                                                   embedding_dim=value_dim,
                                                   padding_idx=PAD_INDEX)
        self._erase_layer = nn.Linear(in_features=value_dim,
                                      out_features=value_dim)
        self._add_layer = nn.Linear(in_features=value_dim,
                                    out_features=value_dim)
        self._summary_layer = nn.Linear(in_features=value_dim+key_dim,
                                        out_features=summary_dim)
        self._output_layer = nn.Linear(in_features=summary_dim,
                                       out_features=1)

        self._key_memory = torch.Tensor(self._concept_num, self._key_dim)
        self._value_memory = torch.Tensor(self._concept_num, self._key_dim)

    def _init_memory(self):
        pass

    def _compute_correlation_weight(self, question_id):
        question_vector = self._question_embeding(question_id)
        return nn.Softmax(torch.matmul(question_vector, self._key_memory), dim=-1)

    def _read(self, question_id):
        correlation_weight = self._compute_correlation_weight(question_id)
        read_content = torch.matmul(correlation_weight, self._value_memory)
        return read_content

    def _write(self, interaction):
        interaction_vector = self._interaction_embedding(interaction)

        self._prev_value_memory = self._value_memory
        self._value_memory = torch.Tensor(self._concept_num, self._value_dim)

        e = nn.Sigmoid(self._erase_layer(interaction_vector))  # erase vector
        a = nn.Tanh(self._add_layer(interaction_vector))  # add vector

        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self._value_memory = self._prev_value_memory * (1 - erase) + add

    def forward(self, input, target_id):
        # initialize memory matrices
        self._init_memory()
        # repeat write process seq_size many times with input
        ...

        # read process
        question_vector = self._question_embeding(target_id)
        read_content = self._read(target_id)
        # TODO: check dimension
        summary_vector = self._summary_layer(torch.cat((read_content, question_vector), dim=-1))
        summary_vector = nn.Tanh(summary_vector)
        output = self._output_layer(summary_vector)
        return output
