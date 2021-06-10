import torch.nn as nn


from FrequencyModulatedKernelBO.experiments.nas.utils import initialize_conv2d, initialize_batchnorm2d, initialize_linear


NASNET_OPS = ['Id',
              'Conv1by1', 'Conv3by3', 'Conv5by5',
              'ConvSeparable3by3', 'ConvSeparable5by5',
              'Maxpool3by3', 'Maxpool5by5']


class _OPS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _OPSConv(_OPS):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=n_channels)
        self.relu = nn.ReLU()

    def initialize_params(self):
        initialize_conv2d(self.conv)
        initialize_batchnorm2d(self.bn)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class _OPSPool(_OPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_params(self):
        pass

    def forward(self, x):
        return self.pool(x)


class OPSConv5by5(_OPSConv):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, padding=2, bias=True)


class OPSConv3by3(_OPSConv):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True)


class OPSConv1by1(_OPSConv):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, bias=True)


class OPSConvSeparable3by3(_OPSConv):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                              kernel_size=3, padding=1, bias=True, groups=4)


class OPSConvSeparable5by5(_OPSConv):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(n_channels, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                              kernel_size=5, padding=2, bias=True, groups=4)


class OPSMaxpool3by3(_OPSPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


class OPSMaxpool5by5(_OPSPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)



# Id can be considered as 1 by 1 max pool
class OPSId(_OPS):
    def __init__(self, *args, **kwargs):
        super(OPSId, self).__init__(*args, **kwargs)

    def initialize_params(self):
        pass

    def forward(self, x):
        return x