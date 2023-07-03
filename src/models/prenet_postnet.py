import torch
import torch.nn as nn
from torch.nn import functional 
from src.models.utilities import LinearTransformation, ConvolutionalNormalization


# the Prenet module applies a series of linear transformations with ReLU activation and dropout to the input tensor.
# It helps extract relevant features from the input before passing it to the main network, providing a non-linear transformation and regularization.

class Prenet(nn.Module):
    def __init__(self, input_dimension, layer_sizes):
        super(Prenet, self).__init__()
        input_sizes = [input_dimension] + list(layer_sizes[:-1])
        self.layers = nn.ModuleList()
        for input_size, out_size in zip(input_sizes, layer_sizes):
            self.layers.append(LinearTransformation(
                input_dim=input_size, output_dim=out_size, bias=False))

    def forward(self, input):
        for layer in self.layers:
            input = functional.dropout(functional.relu(layer(input)), training=True, p=0.5)
        return input


# class represents the postnet module, which is a stack of 1-dimensional convolutions used for post-processing the outputs of a speech synthesis model.
# The postnet helps refine the predicted mel-spectrogram by adding fine-grained details and reducing potential artifacts.


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, config):
        super(Postnet, self).__init__()
        self.convs = nn.ModuleList()
        'The first convolutional layer is defined using ConvNorm and nn.BatchNorm1d.'
        'It takes the mel-spectrogram channels as input (config.number_mel_channels) and applies a 1-dimensional'
        'convolution with config.postnet_dim_emb output channels and a kernel size of config.kernal_size_postnet.'
        'The output of this convolution is then passed through batch normalization.'
        # it makes 5 1-d conv layers with 512 channels , kernel 5
        self.padding = (config.kernal_size_postnet - 1) // 2
        self.convs.append(
            nn.Sequential(
                ConvolutionalNormalization(config.number_mel_channels, config.postnet_dim_emb,
                                           stride=1, weight_init='tanh', padding=self.padding,
                                           kernel_size=config.kernal_size_postnet,
                                           dilation=1),
                nn.BatchNorm1d(config.postnet_dim_emb))
        )

        # For the intermediate convolutions, the same pattern is followed: 1-dimensional convolution, batch normalization, and activation function (tanh)
        rang = config.number_conv_postnet - 2
        for _ in range(2, rang):
            self.convs.append(
                nn.Sequential(
                    ConvolutionalNormalization(config.postnet_dim_emb,
                                               config.postnet_dim_emb, weight_init='tanh',
                                               kernel_size=config.kernal_size_postnet,
                                               padding=self.padding,
                                               stride=1,
                                               dilation=1),
                    nn.BatchNorm1d(config.postnet_dim_emb))
            )

        "The last convolutional layer applies a 1-dimensional convolution with config.postnet_dim_emb input channels and config.number_mel_channels output channels."
        "The kernel size and padding are determined by config.kernal_size_postnet."
        "The activation function used here is linear, and batch normalization is applied."
        self.convs.append(
            nn.Sequential(
                ConvolutionalNormalization(config.postnet_dim_emb, config.number_mel_channels,
                                           stride=1, kernel_size=config.kernal_size_postnet,
                                           padding=self.padding,
                                           dilation=1, weight_init='linear'),
                nn.BatchNorm1d(config.number_mel_channels))
        )
#  forward is applay the multi conv layers takes an input tensor x and processes it through the convolutional layers.

    def forward(self, input):
        for j in range(len(self.convs) - 1):
            #             For each convolutional layer except the last one, the input tensor x is passed through the convolution, followed by the tanh activation function and dropout (functional.dropout).
            input = functional.dropout(torch.tanh(
                self.convs[j](input)), 0.5, self.training)
    #     For the last convolutional layer, only the convolution and dropout are applied, without the tanh activation.
        return functional.dropout(self.convs[-1](input), 0.5, self.training)
