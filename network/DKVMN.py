import torch
import torch.nn as nn
from constant import PAD_INDEX
from config import ARGS
from math import sqrt


class DKVMN(nn.Module):
    """
    Extension of Memory-Augmented Neural Network (MANN)
    """
    def __init__(self, key_dim, value_dim, summary_dim, question_num, concept_num):
        super().__init__()
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._summary_dim = summary_dim
        self._question_num = question_num
        self._concept_num = concept_num

        # embedding layers
        self._question_embedding = nn.Embedding(num_embeddings=question_num+1,
                                                embedding_dim=key_dim,
                                                padding_idx=PAD_INDEX)
        self._interaction_embedding = nn.Embedding(num_embeddings=2*question_num+1,
                                                   embedding_dim=value_dim,
                                                   padding_idx=PAD_INDEX)

        # FC layers
        self._erase_layer = nn.Linear(in_features=value_dim,
                                      out_features=value_dim)
        self._add_layer = nn.Linear(in_features=value_dim,
                                    out_features=value_dim)
        self._summary_layer = nn.Linear(in_features=value_dim+key_dim,
                                        out_features=summary_dim)
        self._output_layer = nn.Linear(in_features=summary_dim,
                                       out_features=1)

        # key memory matrix, transposed and initialized
        self._key_memory = torch.Tensor(self._key_dim, self._concept_num).to(ARGS.device)
        stdev = 1 / (sqrt(self._concept_num + self._key_dim))
        nn.init.uniform_(self._key_memory, -stdev, stdev)

        # activation functions
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()
        self._softmax = nn.Softmax(dim=-1)

    def _transform_interaction_to_question_id(self, interaction):
        """
        get question_id from interaction index
        if interaction index is a number in [0, question_num], then leave it as-is
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        return interaction - self._question_num * (interaction > self._question_num).long()

    def _init_value_memory(self):
        """
        initialize value memory matrix
        follows initialization that used in the following NMT implementation:
        https://github.com/loudinthecloud/pytorch-ntm/blob/master/ntm/memory.py
        """
        # value memory matrix, transposed
        self._value_memory = torch.Tensor(self._value_dim, self._concept_num).to(ARGS.device)

        stdev = 1 / (sqrt(self._concept_num + self._key_dim))
        nn.init.uniform_(self._value_memory, -stdev, stdev)
        self._value_memory = self._value_memory.clone().repeat(self._batch_size, 1, 1)  # (batch_size, key_dim, concept_num)

    def _compute_correlation_weight(self, question_id):
        """
        compute correlation weight of a given question with key memory matrix
        question_id: integer tensor of shape (batch_size)
        """
        question_vector = self._question_embedding(question_id).to(ARGS.device)
        return self._softmax(torch.matmul(question_vector, self._key_memory)).to(ARGS.device)

    def _read(self, question_id):
        """
        read process - get read content vector from question_id and value memory matrix
        question_id: (batch_size)
        """
        question_id = question_id.squeeze(-1)
        correlation_weight = self._compute_correlation_weight(question_id)
        read_content = torch.matmul(self._value_memory, correlation_weight.unsqueeze(-1)).squeeze(-1)
        return read_content.to(ARGS.device)

    def _write(self, interaction):
        """
        write process - update value memory matrix
        interaction: (batch_size)
        """
        interaction_vector = self._interaction_embedding(interaction)
        question_id = self._transform_interaction_to_question_id(interaction)

        self._prev_value_memory = self._value_memory

        e = self._sigmoid(self._erase_layer(interaction_vector))  # erase vector
        a = self._tanh(self._add_layer(interaction_vector))  # add vector

        w = self._compute_correlation_weight(question_id)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        erase = torch.transpose(erase, 1, 2)
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        add = torch.transpose(add, 1, 2)
        self._value_memory = self._prev_value_memory * (1 - erase) + add

    def forward(self, input, target_id):
        """
        get output of the model (before taking sigmoid)
        input: integer tensor of shape (batch_size, sequence_size)
        target_id: integer tensor of shape (batch_size)
        """
        # initialize value memory matrix
        batch_size = input.shape[0]
        self._batch_size = batch_size
        self._init_value_memory()

        # repeat write process seq_size many times with input
        for i in range(ARGS.seq_size):
            interaction = input[:, i]  # (batch_size)
            self._write(interaction)

        # read process
        question_vector = self._question_embedding(target_id)
        question_vector = question_vector.squeeze(1)
        read_content = self._read(target_id)

        summary_vector = self._summary_layer(torch.cat((read_content, question_vector), dim=-1))
        summary_vector = self._tanh(summary_vector)
        output = self._output_layer(summary_vector)
        return output
