import torch
import torch.nn as nn
import math
from src.models.utilities import LinearTransformation, ConvolutionalNormalization


# this for conv it's output is a flatten array
# it's gives same x,y dim but change the number of attention filters to attention dimensions


# this for conv it's output is a flatten array
# it's gives same x,y dim but change the number of attention filters to attention dimensions

# in_channels is the number of attention filters
# out_channels is the number of attention dimensions
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(AttentionConv, self).__init__()
        self.dense_linear_transformation = LinearTransformation(
            in_channels, out_channels, weight_init='tanh', bias=False)
        self.conv = ConvolutionalNormalization(
            2, in_channels, bias=False, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2)

    def forward(self, weights):
        attention = self.conv(weights)
        attention = attention.transpose(1, 2)
        return self.dense_linear_transformation(attention)


class Attention(nn.Module):
    def __init__(self, rnn_hidden_dim, in_emb_dim, out_att_dim,in_channels, kernel_size):
        super(Attention, self).__init__()

        # memory layer
        # A linear layer that transforms the encoder's output (memory) from embedding dim to atten_dim dimensions
        self.encoder_out_layer = LinearTransformation(in_emb_dim, out_att_dim,
                                                      weight_init='tanh', bias=False)
        # query layer
        # A linear layer that transforms the decoder's output (query) from rnn_atten_dim to atten_dim dimensions
        self.decoder_out_layer = LinearTransformation(rnn_hidden_dim, out_att_dim,
                                                      weight_init='tanh', bias=False)
        self.attention_layer = AttentionConv(
            in_channels, out_att_dim, kernel_size)
        # A linear layer that maps the attention dimension (atten_dim) to a scalar value (1) for each time step.
        self.energy_layer = LinearTransformation(out_att_dim, 1, bias=False)
        # Value to be assigned to masked positions in alignment energies
        self.mask_score_value = -math.inf

    # These energies indicate the relevance or importance of each time step in the input sequence for the current decoding step. like nlp example in lec
    def alignment_energies(self,  memory, query, cumulative_attention_weights):
        after_weights = self.attention_layer(
            cumulative_attention_weights)
        after_query = self.decoder_out_layer(query.unsqueeze(1))
        return self.energy_layer(torch.tanh(
            after_query + after_weights + memory)).squeeze(-1)

    # computes the attention context vector and attention weights based on the provided inputs.
    def forward(self, hidden_atten_state, memory, memory_after,
                cumulative_attention_weights,mask):

        energies_alignment = self.alignment_energies(
            memory_after, hidden_atten_state, cumulative_attention_weights)

        if mask is not None:
            energies_alignment.data.masked_fill_(mask, self.mask_score_value)

        weights = torch.softmax(energies_alignment, dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights
