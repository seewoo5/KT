import torch
import torch.nn as nn
from constant import *
from config import ARGS


class FC(nn.Module):
    """
    Last 4 FC layers of NPA
    """
    def __init__(self, user_question_dim, fc_dim):

        super().__init__()
        self._fc1 = nn.Linear(user_question_dim, fc_dim)
        self._fc2 = nn.Linear(fc_dim, fc_dim // 2)
        self._fc3 = nn.Linear(fc_dim // 2, fc_dim // 4)
        self._fc4 = nn.Linear(fc_dim // 4, 1)

        self._relu = nn.ReLU()

        # Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self._fc1(x)
        x = self._relu(x)

        x = self._fc2(x)
        x = self._relu(x)

        x = self._fc3(x)
        x = self._relu(x)

        x = self._fc4(x)
        return x


class NPA(nn.Module):
    """
    Bi-LSTM + Attention based model
    """
    def __init__(self, input_dim=128, hidden_dim=128, attn_dim=256, fc_dim=512,
                 num_layers=1, question_num=QUESTION_NUM[ARGS.dataset_name], dropout=0.0):

        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._question_num = question_num
        self._lstm = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=True)

        # embedding layers
        self._response_embedding_layer = nn.Embedding(num_embeddings=2+1,
                                                      embedding_dim=input_dim,
                                                      padding_idx=PAD_INDEX)
        self._question_embedding_layer = nn.Embedding(num_embeddings=question_num+1,
                                                      embedding_dim=input_dim,
                                                      padding_idx=PAD_INDEX)

        # attention layers
        self._attention_lstm = nn.Linear(in_features=2*hidden_dim, out_features=attn_dim, bias=False)
        self._attention_question = nn.Linear(in_features=input_dim, out_features=attn_dim, bias=False)
        self._attention_weight = nn.Linear(in_features=attn_dim, out_features=1, bias=False)

        # FC layers
        self._fc_layers = FC(2*hidden_dim+input_dim, fc_dim)

        # activation functions
        self._tanh = nn.Tanh()
        self._softmax = nn.Softmax(dim=-1)

        # Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        need to multiply 2 on num_layers since we'are using Bi-LSTM
        """
        weight = next(self.parameters())
        return (weight.new_zeros(2*self._num_layers, batch_size, self._hidden_dim),
                weight.new_zeros(2*self._num_layers, batch_size, self._hidden_dim))

    def _transform_interaction_to_question_id_and_response(self, interaction):
        """
        get question_id and response correctness from interaction index
        if interaction index is a number in [0, question_num], then question_id is same as
        interaction id (except 0, which is a padding), and response correctness
        is 1, which means correct.
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num for question_id, and response correctness
        is 2, which means incorrect.
        For padding (interaction index 0), index for response correctness is also 0
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        question_id = interaction - self._question_num * (interaction > self._question_num).long()
        padding = (question_id == PAD_INDEX)
        response = (interaction <= self._question_num).long()
        response = 2 - response
        response = response * (~padding).long()
        return question_id, response

    def _embedding(self, interaction):
        """
        Embed interactions
        interaction: (batch_Size, sequence_size)
        return interaction_vector, a tensor of shape (batch_size, seq_size, input_dim)
        """
        question_id, response = self._transform_interaction_to_question_id_and_response(interaction)
        question_vector = self._question_embedding_layer(question_id)
        response_vector = self._response_embedding_layer(response)
        return torch.mul(question_vector, response_vector)

    def _attention(self, lstm_output, question_vector):
        """
        Additive attention is used
        lstm_output: (batch_size, sequence_size, 2*hidden_dim)
        return user_vector, a tensor of shape (batch_size, 1, 2*hidden_dim)
        """
        attention_score = self._attention_lstm(lstm_output) + self._attention_question(question_vector)
        attention_score = self._tanh(attention_score)
        attention_score = self._attention_weight(attention_score).squeeze(-1)
        alpha = self._softmax(attention_score).unsqueeze(1)
        return torch.matmul(alpha, lstm_output)

    def forward(self, input, target_id):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, sequence_size)
        target_id: (batch_size)
        return output (response correctness, before taking sigmoid),
        a tensor of shape (batch_size, 1)
        """
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        input = self._embedding(input)
        question_vector = self._question_embedding_layer(target_id)
        # question_vector: (batch_size, 1, input_dim)

        # Bi-LSTM layer
        output, _ = self._lstm(input, (hidden[0].detach(), hidden[1].detach()))
        # output: (batch_size, sequence_size, 2*hidden_dim)

        # Attention layer
        user_vector = self._attention(output, question_vector)
        # user_vector: (batch_size, 2*hidden_dim)

        # FC layers
        user_question_vector = torch.cat([user_vector, question_vector], dim=-1).squeeze(1)
        # user_question_vector: (batch_size, 2*hidden_dim+input_dim)
        output = self._fc_layers(user_question_vector)
        # output: (batch_size, 1)
        return output
