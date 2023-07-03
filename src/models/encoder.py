import torch
import torch.nn as nn
from torch.nn import functional 
from src.models.utilities import  ConvolutionalNormalization


class Encoder(nn.Module):
    
    # BiLSTM and 3 1-d conv layers

    def __init__(self, config):
        super(Encoder, self).__init__()
        padding = (config.kernal_size_encoder - 1) // 2
        convs = []
        for i in range(config.n_convelution_encoder):
            convlution_lay = nn.Sequential(
                ConvolutionalNormalization(input_dim=config.emb_dim_encoder,
                        kernel_size=config.kernal_size_encoder, 
                        padding=padding, stride=1, weight_init='relu',
                    dilation=1, output_dim=config.emb_dim_encoder),
                nn.BatchNorm1d(config.emb_dim_encoder))
            convs.append(convlution_lay)
        self.lstm = nn.LSTM(config.emb_dim_encoder,
                            config.emb_dim_encoder // 2, 1,
                            bidirectional=True,
                            batch_first=True)
        self.convs = nn.ModuleList(convs)


    def forward(self, input, sizes):
        for conv in self.convs:
            input = functional.dropout(functional.relu(conv(input)), 0.5, self.training)

        input = input.permute(0,2, 1)

        sizes = sizes.cpu().numpy()
        input = nn.utils.rnn.pack_padded_sequence(
            input, sizes, batch_first=True)

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)

        return lstm_out

    def inference(self, input):
        for conv in self.convs:
            input = functional.dropout(functional.relu(conv(input)), 0.5, self.training)

        input = input.permute(0,2, 1) 

        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(input)

        return lstm_out
