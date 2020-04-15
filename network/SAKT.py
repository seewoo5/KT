import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import PAD_INDEX
from config import ARGS
from network.util_network import get_pad_mask, get_subsequent_mask


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class SAKT(nn.Module):
    """
    Transformer-based
    all hidden dimensions (d_k, d_v, ...) are the same as hidden_dim
    single attention block
    """
    def __init__(self, hidden_dim, question_num, num_layers, num_head, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num

        # self-attention & feed-forward networks
        self._self_attn = MultiHeadAttention(num_head, hidden_dim, hidden_dim, hidden_dim, dropout)
        self._ffn = PositionwiseFeedForward(hidden_dim, hidden_dim, dropout)

        # prediction layer
        self._prediction = nn.Linear(hidden_dim, 1)

        # Embedding layers
        self._positional_embedding = nn.Embedding(ARGS.seq_size+1, hidden_dim, padding_idx=PAD_INDEX)
        self._interaction_embedding = nn.Embedding(2*question_num+1, hidden_dim, padding_idx=PAD_INDEX)
        self._question_embedding = nn.Embedding(question_num+1, hidden_dim, padding_idx=PAD_INDEX)

    def _transform_interaction_to_question_id(self, interaction):
        """
        get question_id from interaction index
        if interaction index is a number in [0, question_num], then leave it as-is
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        return interaction - self._question_num * (interaction > self._question_num).long()

    def _get_position_index(self, question_id):
        """
        [0, 0, 0, 4, 12] -> [0, 0, 0, 1, 2]
        """
        batch_size = question_id.shape[0]
        position_indices = []
        for i in range(batch_size):
            non_padding_num = (question_id[i] != PAD_INDEX).sum(-1).item()
            position_index = [0] * (ARGS.seq_size - non_padding_num) + list(range(1, non_padding_num+1))
            position_indices.append(position_index)
        return torch.tensor(position_indices, dtype=int).to(ARGS.device)

    def forward(self, interaction_id, target_id):
        """
        Query: Question (skill, exercise, ...) embedding
        Key, Value: Interaction embedding + positional embedding
        """
        question_id = self._transform_interaction_to_question_id(interaction_id)
        question_id = torch.cat([question_id[:, 1:], target_id], dim=-1)

        interaction_vector = self._interaction_embedding(interaction_id)
        question_vector = self._question_embedding(question_id)
        position_index = self._get_position_index(question_id)
        position_vector = self._positional_embedding(position_index)

        # TODO: subsequent mask
        mask = get_pad_mask(question_id, PAD_INDEX) & get_subsequent_mask(question_id)
        output, _ = self._self_attn(q=question_vector,
                                 k=interaction_vector+position_vector,
                                 v=interaction_vector+position_vector,
                                 mask=mask)
        output = self._ffn(output)
        output = self._prediction(output)
        output = output[:, -1, :]
        return output
