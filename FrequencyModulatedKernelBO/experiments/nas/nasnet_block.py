from typing import List, Tuple

import torch.nn as nn


from FrequencyModulatedKernelBO.experiments.nas.nasnet_ops import NASNET_OPS, _OPSConv, \
    OPSId, OPSConv1by1, OPSConv3by3, OPSConv5by5, \
    OPSConvSeparable3by3, OPSConvSeparable5by5, OPSMaxpool3by3, OPSMaxpool5by5


class NASNetBlock(nn.Module):
    def __init__(self, block_architecture: List[Tuple[Tuple[int, str], Tuple[int, str]]],
                 block_output_state: int, n_channel: int):
        super().__init__()
        for i, elm in enumerate(block_architecture):
            (input1, op1), (input2, op2) = elm
            assert input1 < input2 < len(block_architecture)
            if i == 0:
                assert input1 == -1 and input2 == 0
            if i == len(block_architecture):
                assert not (input1 == -1 and input2 == 0)
            assert op1 in NASNET_OPS and op2 in NASNET_OPS
        assert 1 <= block_output_state < len(block_architecture)
        self._block_architecture = block_architecture
        self._block_output_state = block_output_state
        for i, elm in enumerate(self._block_architecture):
            (input1, op1), (input2, op2) = elm
            module1 = globals()['OPS' + op1]
            module2 = globals()['OPS' + op2]
            setattr(self, 'state%d-op1' % (i + 1), module1(n_channel))
            setattr(self, 'state%d-op2' % (i + 1), module2(n_channel))

    def forward(self, x1, x2):
        activation_dict = {-1: x1, 0: x2}
        for i, elm in enumerate(self._block_architecture):
            (input1, op1), (input2, op2) = elm
            activation1 = getattr(self, 'state%d-op1' % (i + 1))(activation_dict[input1])
            activation2 = getattr(self, 'state%d-op2' % (i + 1))(activation_dict[input2])
            activation_dict[i + 1] = activation1 + activation2
        last_activation_ind = len(self._block_architecture)
        return activation_dict[self._block_output_state], activation_dict[last_activation_ind]

    def initialize_params(self):
        for i, elm in enumerate(self._block_architecture):
            getattr(self, 'state%d-op1' % (i + 1)).initialize_params()
            getattr(self, 'state%d-op2' % (i + 1)).initialize_params()