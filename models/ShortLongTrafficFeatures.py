import torch.nn as nn


class TrafficFeatureLSTM (nn.Module):

    def __init__(self):
        super(TrafficFeatureLSTM, self).__init__()
        self.build()

    def build(self):
        pass

    def forward(self, *input):
        pass


class ShortTermLSTM(TrafficFeatureLSTM):

    def __init__(self, d_n):
        super(TrafficFeatureLSTM, self).__init__()
        self.d_n = d_n

    def forward(self, temporal, spatial, short_ttf):
        pass


class LongTermLSTM(TrafficFeatureLSTM):
    pass
