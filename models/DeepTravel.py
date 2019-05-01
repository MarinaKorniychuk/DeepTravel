import torch
import torch.nn as nn

from models.ShortLongTrafficFeatures import  ShortTermLSTM, LongTermLSTM
from models.SpatialTemporal import SpatialTemporal


class DeepTravel(nn.Module):
    def __init__(self, alpha=0.3):
        super(DeepTravel, self).__init__()

        self.alpha = alpha
        self.build()
        self.init_weight()

    def build(self):
        self.spatial_temporal = SpatialTemporal()
        self.short_term_lstm = ShortTermLSTM()
        self.long_term_lstm = LongTermLSTM()

    def evaluate(self, stats, temporal, spatial, dr_state, short_ttf=None, long_ttf=None):
        self(stats, temporal, spatial, dr_state, short_ttf, long_ttf)

    def forward(self, stats, temporal, spatial, dr_state, short_ttf, long_ttf):
        V_sp, V_tp = self.spatial_temporal(stats, temporal, spatial)

        V_dri = dr_state['dr_state']

        V_short = self.short_term_lstm(short_ttf)
        V_long = self.long_term_lstm(long_ttf)

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.uniform(param.data)
