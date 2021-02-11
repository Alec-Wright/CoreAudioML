import CoreAudioML.dataset as dataset
import CoreAudioML.training as training
import CoreAudioML.networks as networks
import os
import torch
import math
import numpy as np


def test_system():
    network = networks.RecNet()
    network.add_layer({'block_type': 'GRU', 'input_size': 2, 'output_size': 1, 'hidden_size': 16})

    data = dataset.DataSet(os.path.join('result_test', 'network1'))
    data.create_subset('train', 8820)
    data.load_file('KDonnerFlangerra12c12rg9Singles1', set_names='train')

    optimiser = torch.optim.Adam(network.parameters(), lr=0.0001)
    loss_fcn = training.ESRLoss()
    batch_size = 10
    init_len = 120
    up_fr = 100
    for i in range(3):
        print('Starting epoch ' + str(i+1))
        train_losses = []
        # shuffle the segments at the start of the epoch
        train_input = data.subsets['train'].data['input'][0]
        train_target = data.subsets['train'].data['target'][0]
        shuffle = torch.randperm(train_input.shape[1])

        for n in range(math.ceil(shuffle.shape[0]/batch_size)):
            # Load batch of randomly selected segments
            batch_losses = []
            batch = train_input[:, shuffle[n*batch_size:(n+1)*batch_size], :]
            target = train_target[:, shuffle[n*batch_size:(n+1)*batch_size], :]

            # Run initialisation samples through the network
            network(batch[0:init_len, :, :])
            # Zero the gradient buffers
            network.zero_grad()

            # Iterate over the remaining samples in the sequence batch
            for k in range(math.ceil((batch.shape[0]-init_len)/up_fr)):
                output = network(batch[init_len+(k*up_fr):init_len+((k+1)*up_fr), :, :])
                # Calculate loss
                loss = loss_fcn(output, target[init_len+(k*up_fr):init_len+((k+1)*up_fr), :, :])
                train_losses.append(loss.item())
                batch_losses.append(loss.item())

                loss.backward()
                optimiser.step()

                # Set the network hidden state, to detach it from the computation graph
                network.detach_hidden()

            print('batch ' + str(n+1) + ' loss = ' + str(np.mean(batch_losses)))
        print('epoch ' + str(i+1) + ' loss = ' + str(np.mean(train_losses)))

