import torch 


# this class for applying linear transformation for the data 

class LinearTransformation(torch.nn.Module):
  def __init__(self, input_dim, output_dim, weight_init='linear', bias=True):
      super(LinearTransformation, self).__init__()
      self.linear_lay = torch.nn.Linear(input_dim, output_dim, bias=bias)

      # Initialize the weight using Xavier uniform initialization
      torch.nn.init.xavier_uniform_(
          self.linear_lay.weight,
          gain=torch.nn.init.calculate_gain(weight_init))

  def forward(self, input_data):
      # Apply linear transformation to the input data
      # print(input_data.shape)
      return self.linear_lay(input_data)


'''The ConvolutionalNormalization class is responsible for applying a 1-dimensional convolutional
operation on the input signal, followed by normalization. using it 
enabling effective feature extraction and pattern recognition from input signals.'''

# The normalization step helps stabilize and enhance the learning process by scaling the values within the convolutional layer.


class ConvolutionalNormalization(torch.nn.Module):
  def __init__(self, input_dim, output_dim, weight_init='linear', bias=True,
              kernel_size=1, stride=1,padding=None, dilation=1):
    super(ConvolutionalNormalization, self).__init__()
    

    if padding == None :
      # for insure symetric padding
        assert kernel_size % 2 == 1, "Kernel size must be odd for automatic padding"
        padding = dilation * (kernel_size - 1) 
        padding = padding // 2

    # create a convolutional layer instance from the input parameters
    self.conv1d = torch.nn.Conv1d(input_dim, output_dim,  bias=bias, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation)

    # Initialize the weights of conv layer using Xavier uniform initialization
    torch.nn.init.xavier_uniform_(
        self.conv1d.weight, gain=torch.nn.init.calculate_gain(weight_init))

  def forward(self, input):
    return self.conv1d(input)

