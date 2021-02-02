import torch
import torch.nn as nn
import networks


def test_net(network):
    for n in range(3):
        output = network(torch.ones([1000, 10, network.input_size]))
        target = torch.empty([1000, 10, network.output_size])

        assert output.size() == target.size()

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        loss = torch.mean(torch.pow(torch.add(output, -target), 2))

        loss.backward()
        optimizer.step()
        network.detach_hidden()


if __name__ == "__main__":
    """Creates and tests the various neural network models"""

    block_params1 = {'layer_name': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
    block_params2 = {'layer_name': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
    block_params3 = {'layer_name': 'LSTM', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}

    block_params4 = {'layer_name': 'LSTM', 'input_size': 2, 'output_size': 4, 'hidden_size': 16}
    block_params5 = {'layer_name': 'GRU', 'input_size': 4, 'output_size': 8, 'hidden_size': 16}

    try:
        test_network1 = networks.RecNet(block_params1)
        test_net(test_network1)
    except:
        print('test network1 failed')

    try:
        test_network2 = networks.RecNet([block_params1, block_params2, block_params3])
        test_net(test_network2)
    except:
        print('test network2 failed')

    try:
        test_network3 = networks.RecNet(None)
        test_network3.add_layer(block_params4)
        test_network3.add_layer(block_params5)
    except:
        print('test network3 failed')





