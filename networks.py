import torch
import torch.nn as nn
import CoreAudioML.miscfuncs as miscfuncs
import math
from contextlib import nullcontext

def wrapperkwargs(func, kwargs):
    return func(**kwargs)

def wrapperargs(func, args):
    return func(*args)

# A simple RNN class that consists of a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
# layer

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1, bias_fl=True,
                 num_layers=1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Create dictionary of possible block types
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.skip = skip
        self.save_state = True
        self.hidden = None

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, 0:self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'SimpleRNN', 'input_size': self.rec.input_size, 'skip': self.skip,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss


""" 
Gated Convolutional Neural Net class, based on the 'WaveNet' architecture, takes a single channel of audio as input and
produces a single channel of audio of equal length as output. one-sided zero-padding is used to ensure the network is 
causal and doesn't reduce the length of the audio.

Made up of 'blocks', each one applying a series of dilated convolutions, with the dilation of each successive layer 
increasing by a factor of 'dilation_growth'. 'layers' determines how many convolutional layers are in each block,
'kernel_size' is the size of the filters. Channels is the number of convolutional channels.

The output of the model is creating by the linear mixer, which sums weighted outputs from each of the layers in the 
model

"""


class GatedConvNet(nn.Module):
    def __init__(self, channels=8, blocks=2, layers=9, dilation_growth=2, kernel_size=3):
        super(GatedConvNet, self).__init__()
        # Set number of layers  and hidden_size for network layer/s
        self.layers = layers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.save_state = True
        for b in range(blocks):
            self.blocks.append(ResConvBlock1DCausalGated(1 if b == 0 else channels, channels, dilation_growth,
                                                         kernel_size, layers))
        self.blocks.append(nn.Conv1d(channels*layers*blocks, 1, 1, 1, 0))

    def forward(self, x):
        x = x.permute(1, 2, 0)
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])
        for n, block in enumerate(self.blocks[:-1]):
            x, zn = block(x)
            z[:, n*self.channels*self.layers:(n + 1) * self.channels*self.layers, :] = zn
        return self.blocks[-1](z).permute(2, 0, 1)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            self.zero_grad()
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Process batch
            output = self(input_batch)

            # Calculate loss and update network parameters
            loss = loss_fcn(output, target_batch)
            loss.backward()
            optim.step()

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += loss

        return ep_loss / (batch_i + 1)

    # only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = self(input_data)
            loss = loss_fcn(output, target_data)
        return output, loss

    # This functions saves the model and all its parameters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'GatedConvNet', 'layers': self.layers, 'channels': self.channels,
                                     'dilation_growth': self.dilation_growth, 'kernel_size': self.kernel_size,
                                     'blocks': len(self.blocks) - 1}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)


""" 
Gated convolutional neural net block, applies successive gated convolutional layers to the input, a total of 'layers'
layers are applied, with the filter size 'kernel_size' and the dilation increasing by a factor of 'dilation_growth' for
 each successive layer."""


class ResConvBlock1DCausalGated(nn.Module):
    def __init__(self, chan_input, chan_output, dilation_growth, kernel_size, layers):
        super(ResConvBlock1DCausalGated, self).__init__()
        self.channels = chan_output

        dilations = [dilation_growth ** lay for lay in range(layers)]
        self.layers = nn.ModuleList()

        for dil in dilations:
            self.layers.append(ResConvLayer1DCausalGated(chan_input, chan_output, dil, kernel_size))
            chan_input = chan_output

    def forward(self, x):
        z = torch.empty([x.shape[0], len(self.layers)*self.channels, x.shape[2]])
        for n, layer in enumerate(self.layers):
            x, zn = layer(x)
            z[:, n*self.channels:(n + 1) * self.channels, :] = zn
        return x, z


""" 
Gated convolutional layer, zero pads and then applies a causal convolution to the input """


class ResConvLayer1DCausalGated(nn.Module):

    def __init__(self, chan_input, chan_output, dilation, kernel_size):
        super(ResConvLayer1DCausalGated, self).__init__()
        self.channels = chan_output

        self.conv = nn.Conv1d(in_channels=chan_input, out_channels=chan_output * 2, kernel_size=kernel_size, stride=1,
                              padding=0, dilation=dilation)
        self.mix = nn.Conv1d(in_channels=chan_output, out_channels=chan_output, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        z = torch.tanh(y[:, 0:self.channels, :]) * torch.sigmoid(y[:, self.channels:, :])

        # Zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0], self.channels, residual.shape[2]-z.shape[2]), z), dim=2)
        x = self.mix(z) + residual
        return x, z


""" 
Recurrent Neural Network class, blocks is a list of layers, each layer is described by a dictionary, layers can also
be added after initialisation via the 'add_layer' function

params is a dict that holds 'meta parameters' for the whole network
skip inserts a skip connection from the input to the output, the value of skip determines how many of the input
channels to add to the output (if skip = 2, for example, the output must have at least two channels)

e.g blocks = {'block_type': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}

This allows you to add an arbitrary number of RNN blocks. The SimpleRNN is easier to use but only includes one reccurent
unit followed by a fully connect layer.
"""
class RecNet(nn.Module):
    def __init__(self, blocks=None, skip=0):
        super(RecNet, self).__init__()
        if type(blocks) == dict:
            blocks = [blocks]
        # Create container for layers
        self.layers = nn.Sequential()
        # Create dictionary of possible block types
        self.block_types = {}
        self.block_types.update(dict.fromkeys(['RNN', 'LSTM', 'GRU'], BasicRNNBlock))
        self.skip = skip
        self.save_state = False
        self.input_size = None
        self.training_info = {'current_epoch': 0, 'training_losses': [], 'validation_losses': [],
                              'train_epoch_av': 0.0, 'val_epoch_av': 0.0, 'total_time': 0.0, 'best_val_loss': 1e12}
        # If layers were specified, create layers
        try:
            for each in blocks:
                self.add_layer(each)
        except TypeError:
            print('no blocks provided, add blocks to the network via the add_layer method')

    # Define forward pass
    def forward(self, x):
        if not self.skip:
            return self.layers(x)
        else:
            res = x[:, :, 0:self.skip]
            return self.layers(x) + res

    # Set hidden state to specified values, resets gradient tracking
    def detach_hidden(self):
        for each in self.layers:
            each.detach_hidden()

    def reset_hidden(self):
        for each in self.layers:
            each.reset_hidden()
            
    # Add layer to the network, params is a dictionary contains the layer keyword arguments
    def add_layer(self, params):
        # If this is the first layer, define the network input size
        if self.input_size:
            pass
        else:
            self.input_size = params['input_size']

        self.layers.add_module('block_'+str(1 + len(list(self.layers.children()))),
                               self.block_types[params['block_type']](params))
        self.output_size = params['output_size']

    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)

        model_data = {'model_data': {'model': 'RecNet', 'skip': 0}, 'blocks': {}}
        for i, each in enumerate(self.layers):
            model_data['blocks'][str(i)] = each.params

        if self.training_info:
            model_data['training_info'] = self.training_info

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)


class BasicRNNBlock(nn.Module):
    def __init__(self, params):
        super(BasicRNNBlock, self).__init__()
        assert type(params['input_size']) == int, "an input_size of int type must be provided in 'params'"
        assert type(params['output_size']) == int, "an output_size of int type must be provided in 'params'"
        assert type(params['hidden_size']) == int, "an hidden_size of int type must be provided in 'params'"

        rec_params = {i: params[i] for i in params if i in ['input_size', 'hidden_size', 'num_layers']}
        self.params = params
        # This just calls nn.LSTM() if 'block_type' is LSTM, nn.GRU() if GRU, etc
        self.rec = wrapperkwargs(getattr(nn, params['block_type']), rec_params)
        self.lin_bias = params['lin_bias'] if 'lin_bias' in params else False
        self.lin = nn.Linear(params['hidden_size'], params['output_size'], self.lin_bias)
        self.hidden = None
        # If the 'skip' param was provided, set to provided value (1 for skip connection, 0 otherwise), is 1 by default
        if 'skip' in params:
            self.skip = params['skip']
        else:
            self.skip = 1

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, 0:self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    def reset_hidden(self):
        self.hidden = None


def load_model(model_data):
    model_types = {'RecNet': RecNet, 'SimpleRNN': SimpleRNN, 'GatedConvNet': GatedConvNet}

    model_meta = model_data.pop('model_data')

    if model_meta['model'] == 'SimpleRNN' or model_meta['model'] == 'GatedConvNet':
        network = wrapperkwargs(model_types[model_meta.pop('model')], model_meta)
        if 'state_dict' in model_data:
            state_dict = network.state_dict()
            for each in model_data['state_dict']:
                state_dict[each] = torch.tensor(model_data['state_dict'][each])
            network.load_state_dict(state_dict)

    elif model_meta['model'] == 'RecNet':
        model_meta['blocks'] = []
        network = wrapperkwargs(model_types[model_meta.pop('model')], model_meta)
        for i in range(len(model_data['blocks'])):
            network.add_layer(model_data['blocks'][str(i)])

        # Get the state dict from the newly created model and load the saved states, if states were saved
        if 'state_dict' in model_data:
            state_dict = network.state_dict()
            for each in model_data['state_dict']:
                state_dict[each] = torch.tensor(model_data['state_dict'][each])
            network.load_state_dict(state_dict)

        if 'training_info' in model_data.keys():
            network.training_info = model_data['training_info']

    return network


# This is a function for taking the old json config file format I used to use and converting it to the new format
def legacy_load(legacy_data):
    if legacy_data['unit_type'] == 'GRU' or legacy_data['unit_type'] == 'LSTM':
        model_data = {'model_data': {'model': 'RecNet', 'skip': 0}, 'blocks': {}}
        model_data['blocks']['0'] = {'block_type': legacy_data['unit_type'], 'input_size': legacy_data['in_size'],
                                     'hidden_size': legacy_data['hidden_size'],'output_size': 1, 'lin_bias': True}
        if legacy_data['cur_epoch']:
            training_info = {'current_epoch': legacy_data['cur_epoch'], 'training_losses': legacy_data['tloss_list'],
                             'val_losses': legacy_data['vloss_list'], 'load_config': legacy_data['load_config'],
                             'low_pass': legacy_data['low_pass'], 'val_freq': legacy_data['val_freq'],
                             'device': legacy_data['pedal'], 'seg_length': legacy_data['seg_len'],
                             'learning_rate': legacy_data['learn_rate'], 'batch_size': legacy_data['batch_size'],
                             'loss_func': legacy_data['loss_fcn'], 'update_freq': legacy_data['up_fr'],
                             'init_length': legacy_data['init_len'], 'pre_filter': legacy_data['pre_filt']}
            model_data['training_info'] = training_info

        if 'state_dict' in legacy_data:
            state_dict = legacy_data['state_dict']
            state_dict = dict(state_dict)
            new_state_dict = {}
            for each in state_dict:
                new_name = each[0:7] + 'block_1.' + each[9:]
                new_state_dict[new_name] = state_dict[each]
            model_data['state_dict'] = new_state_dict
        return model_data
    else:
        print('format not recognised')
