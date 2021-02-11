import CoreAudioML.networks as networks
import CoreAudioML.dataset as dataset
import torch
import os
import miscfuncs
import CoreAudioML.training as training


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


class TestRecNet:
    def test_forward(self):
        block_params1 = {'block_type': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'block_type': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params3 = {'block_type': 'LSTM', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}

        network1 = networks.RecNet(block_params1)
        run_net(network1)

        network2 = networks.RecNet([block_params1, block_params2, block_params3])
        run_net(network2)

        block_params4 = {'block_type': 'RNN', 'input_size': 4, 'output_size': 8, 'hidden_size': 16, 'skip': 0}
        block_params5 = {'block_type': 'RNN', 'input_size': 8, 'output_size': 16, 'hidden_size': 16, 'skip': 0}
        network3 = networks.RecNet([block_params4, block_params5])
        run_net(network3)

        network4 = networks.RecNet({'block_type': 'RNN', 'input_size': 4, 'output_size': 8, 'hidden_size': 8, 'skip':0},
                                   skip=1)
        run_net(network4)

    def test_detach_hidden(self):
        block_params1 = {'block_type': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'block_type': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet([block_params1, block_params2])
        network(torch.ones([100, 10, network.input_size]))
        hidden = []
        for each in network.layers:
            hidden.append(each.hidden)
        network.detach_hidden()
        for i, each in enumerate(network.layers):
            assert torch.all(torch.eq(hidden[i], each.hidden))

    def test_add_layer(self):
        block_params1 = {'block_type': 'LSTM', 'input_size': 4, 'output_size': 2, 'hidden_size': 16}
        block_params2 = {'block_type': 'GRU', 'input_size': 2, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet(None)
        network.add_layer(block_params1)
        network.add_layer(block_params2)
        run_net(network)

    def test_save_model_struct(self):
        block_params1 = {'block_type': 'RNN', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        block_params2 = {'block_type': 'GRU', 'input_size': 1, 'output_size': 1, 'hidden_size': 16}
        network = networks.RecNet([block_params1, block_params2])
        output1 = network(torch.ones([100, 10, network.input_size]))

        network.save_model('TestModel_Stateless', 'test_model')
        network.save_state = True
        network.save_model('TestModel_Stateful', 'test_model')

        del network
        model_data1 = miscfuncs.json_load(os.path.join('test_model', 'TestModel_Stateful'))
        network = networks.load_model(model_data1)
        output2 = network(torch.ones([100, 10, network.input_size]))
        assert torch.all(torch.eq(output1, output2))

        del network
        model_data2 = miscfuncs.json_load(os.path.join('test_model', 'TestModel_Stateless'))
        network = networks.load_model(model_data2)
        output3 = network(torch.ones([100, 10, network.input_size]))
        assert not torch.all(torch.eq(output1, output3))

        os.remove(os.path.join("test_model", "TestModel_Stateful.json"))
        os.remove(os.path.join("test_model", "TestModel_Stateless.json"))
        os.rmdir('test_model')

    def test_network_loading(self):
        network_params = miscfuncs.json_load(os.path.join('result_test', 'network1', 'config.json'))
        network_params['state_dict'] = torch.load(os.path.join('result_test', 'network1', 'modelBest.pt'),
                                                  map_location=torch.device('cpu'))
        model_data = networks.legacy_load(network_params)

        network = networks.load_model(model_data)

        data = dataset.DataSet(os.path.join('result_test', 'network1'))
        data.create_subset('test', 0)
        data.load_file('KDonnerFlangerra12c12rg9Singles1', set_names='test')

        with open(os.path.join('result_test', 'network1', 'tloss.txt')) as fp:
            x = fp.read()
        with torch.no_grad():
            output = network(data.subsets['test'].data['input'][0])
            loss_fcn = training.ESRLoss()

            loss = loss_fcn(output, data.subsets['test'].data['target'][0])
            assert abs(loss.item() - float(x[1:-1])) < 1e-5



