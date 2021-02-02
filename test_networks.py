import networks
import torch
import os

def run_net(network):
    for n in range(3):
        output = network(torch.ones([100, 10, network.input_size]))
        target = torch.empty([100, 10, network.output_size])

        assert output.size() == target.size()

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        loss = torch.mean(torch.pow(torch.add(output, -target), 2))
        loss.backward()
        optimizer.step()
        network.detach_hidden()


class Test_RecNet:
    def test_forward(self):
        block_params1 = {'layer_name': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'layer_name': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params3 = {'layer_name': 'LSTM', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}

        network1 = networks.RecNet(block_params1)
        run_net(network1)

        network2 = networks.RecNet([block_params1, block_params2, block_params3])
        run_net(network2)

        block_params4 = {'layer_name': 'RNN', 'input_size': 4, 'output_size': 8, 'hidden_size': 16, 'skip': 0}
        block_params5 = {'layer_name': 'RNN', 'input_size': 8, 'output_size': 16, 'hidden_size': 16, 'skip': 0}
        network3 = networks.RecNet([block_params4, block_params5])
        run_net(network3)

        network4 = networks.RecNet(skip=1)
        run_net(network4)

    def test_detach_hidden(self):
        block_params1 = {'layer_name': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'layer_name': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet([block_params1, block_params2])
        network(torch.ones([100, 10, network.input_size]))
        hidden = []
        for each in network.layers:
            hidden.append(each.hidden)
        network.detach_hidden()
        for i, each in enumerate(network.layers):
            assert torch.all(torch.eq(hidden[i], each.hidden))

    def test_add_layer(self):
        block_params1 = {'layer_name': 'LSTM', 'input_size': 4, 'output_size': 2, 'hidden_size': 16}
        block_params2 = {'layer_name': 'GRU', 'input_size': 2, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet(None)
        network.add_layer(block_params1)
        network.add_layer(block_params2)
        run_net(network)

    def test_save_model_struct(self):
        block_params1 = {'layer_name': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'layer_name': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet([block_params1, block_params2])
        output1 = network(torch.ones([100, 10, network.input_size]))

        network.save_model('TestModel', 'test_model')

        del network

        network = networks.load_model('TestModel', 'test_model')
        output2 = network(torch.ones([100, 10, network.input_size]))

        assert torch.all(torch.eq(output1, output2))

        os.remove("test_model/TestModel.json")
        os.rmdir('test_model')
