import CoreAudioML.training as training
import torch


class TestLosses:
    def test_ESR(self):
        loss_fcn = training.ESRLoss()
        output = torch.zeros([100, 10, 5], requires_grad=True)
        target = torch.ones([100, 10, 5], requires_grad=True)

        loss = loss_fcn(output, target)
        loss.backward()

        loss = loss_fcn(output, output)
        assert loss.item() == 0

    def test_DC(self):
        loss_fcn = training.DCLoss()
        output = torch.zeros([100, 10, 5], requires_grad=True)
        target = torch.ones([100, 10, 5], requires_grad=True)

        loss = loss_fcn(output, target)
        loss.backward()

        loss = loss_fcn(output, output)
        assert loss.item() == 0

    def test_pre_emph(self):
        loss_fcn = training.ESRLoss()
        pre_emph = training.PreEmph([0.95, 1])
        output = torch.zeros([100, 10, 1], requires_grad=True)
        target = torch.ones([100, 10, 1], requires_grad=True)

        output_pre, target_pre = pre_emph(output, target)

        assert output.shape == output_pre.shape
        assert target.shape == target_pre.shape

        loss = loss_fcn(output_pre, target_pre)
        loss.backward()
