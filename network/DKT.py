import torch
import torch.nn as nn
from constant import PAD_INDEX


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_question, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._encoder = nn.Embedding(num_embeddings=2*num_question+1, embedding_dim=input_dim, padding_idx=PAD_INDEX)
        self._decoder = nn.Linear(hidden_dim, num_question)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self._num_layers, batch_size, self._hidden_dim),
                weight.new_zeros(self._num_layers, batch_size, self._hidden_dim))

    def forward(self, input):
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        input = self._encoder(input)
        output, hidden = self._lstm(input, (hidden[0].detach(), hidden[1].detach()))
        output = self._decoder(output[:,-1,:])
        return output
