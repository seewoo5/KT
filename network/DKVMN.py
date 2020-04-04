# TODO
import torch
import torch.nn as nn
from constant import PAD_INDEX
from config import args


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

    def _transform_interaction_to_question_id(self, interaction):
        """
        get question_id from interaction index
        if interaction index is a number in [0, question_num], then leave it as-is
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        return interaction - self._question_num * (interaction >= self._question_num).long()

    def _init_memory(self):
        """
        initialize key and value memory matrices
        """
        pass

    def _compute_correlation_weight(self, question_id):
        """
        compute correlation weight of a given question with key memory matrix
        question_id: integer tensor of shape (batch_size)
        """
        question_vector = self._question_embeding(question_id)
        return nn.Softmax(torch.matmul(question_vector, self._key_memory), dim=-1)

    def _read(self, question_id):
        """
        read process - get read content vector from question_id and value memory matrix
        question_id: (batch_size)
        """
        correlation_weight = self._compute_correlation_weight(question_id)
        read_content = torch.matmul(correlation_weight, self._value_memory)
        return read_content

    def _write(self, interaction):
        """
        write process - update value memory matrix
        interaction: (batch_size)
        """
        interaction_vector = self._interaction_embedding(interaction)
        question_id = self._transform_interaction_to_question_id(interaction)

        self._prev_value_memory = self._value_memory
        self._value_memory = torch.Tensor(self._concept_num, self._value_dim)

        e = nn.Sigmoid(self._erase_layer(interaction_vector))  # erase vector
        a = nn.Tanh(self._add_layer(interaction_vector))  # add vector

        w = self._compute_correlation_weight(question_id)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self._value_memory = self._prev_value_memory * (1 - erase) + add

    def forward(self, input, target_id):
        """
        get output of the model (before taking sigmoid)
        input: integer tensor of shape (batch_size, sequence_size)
        target_id: integer tensor of shape (batch_size)
        """
        # initialize memory matrices
        self._init_memory()

        # repeat write process seq_size many times with input
        for i in range(args.seq_size):
            interaction = input[:, i]  # (batch_size)
            self._write(interaction)

        # read process
        question_vector = self._question_embeding(target_id)
        read_content = self._read(target_id)

        summary_vector = self._summary_layer(torch.cat((read_content, question_vector), dim=-1))
        summary_vector = nn.Tanh(summary_vector)
        output = self._output_layer(summary_vector)
        return output
