import torch
import torch.nn as nn

"""This is a class for a recurrent neural network type, it is constructed from
blocks specified by blocks argument, which is a list of lists. Each item in 
the list contains the name of the block type, then a list of the blocks
input size, output size, and a list of any other parameters (which vary 
depending on the block type)"""


def wrapper(func, kwargs):
    return func(**kwargs)


# Recurrent Neural Network class, blocks is a list of layers, each layer is described by a dictionary
class RecNet(nn.Module):
    def __init__(self, blocks=[{'layer_name': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}]):
        super(RecNet, self).__init__()
        if type(blocks) == dict:
            blocks = [blocks]
        # Create container for layers
        self.layers = torch.nn.Sequential()
        # Create dictionary of possible block types
        self.block_types = {}
        self.block_types.update(dict.fromkeys(['RNN', 'LSTM', 'GRU'], BasicRNNBlock))
        # If layers were specified, create layers
        try:
            for each in blocks:
                self.add_layer(each)
        except TypeError:
            print('no blocks provided, add blocks to the network via the add_layer method')

    # Define forward pass
    def forward(self, x):
        return self.layers(x)

    # Set hidden state to specified values, resets gradient tracking
    def detach_hidden(self):
        for each in self.layers:
            each.detach_hidden()
            
    # Add layer to the network, params is a dictionary contains the layer keyword arguments
    def add_layer(self, params):
        # If this is the first layer, define the network input size
        try:
            self.input_size
        except torch.nn.modules.module.ModuleAttributeError:
            self.input_size = params['input_size']

        self.layers.add_module('block_'+str(1 + len(list(self.layers.children()))),
                               self.block_types[params['layer_name']](params))
        self.output_size = params['output_size']


class BasicRNNBlock(nn.Module):
    def __init__(self, params):
        super(BasicRNNBlock, self).__init__()
        assert type(params['input_size']) == int, "an input_size of int type must be provided in 'params'"
        assert type(params['output_size']) == int, "an output_size of int type must be provided in 'params'"
        assert type(params['hidden_size']) == int, "an hidden_size of int type must be provided in 'params'"

        rec_params = {i: params[i] for i in params if i in ['input_size', 'hidden_size', 'num_layers']}
        self.rec = wrapper(getattr(nn, params['layer_name']), rec_params)
        self.lin = nn.Linear(params['hidden_size'], params['output_size'])
        self.hidden = None

    def forward(self, x):
        res = x[:, :, 0:self.lin.out_features]
        x, self.hidden = self.rec(x, self.hidden)
        return self.lin(x) + res

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

