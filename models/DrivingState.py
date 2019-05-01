import torch
import torch.nn as nn


class DrivingState(nn.Module):
    """Generate Vdri vector (e.i [1, 0, 0, 0.1]) for each cell in the path."""

    emb_dim = [('dr_state', 16000, 5)]

    def __init__(self):
        super(DrivingState, self).__init__()
        self.build()

    def build(self):
        for name, dim_in, dim_out in DrivingState.emb_dim:
            self.add_module(name + '_em', nn.Embedding(16000, 5))

    def forward(self, dr_state):
        V_dri = []
        for name, dim_in, dim_out in DrivingState.emb_dim:
            embed = getattr(self, name + '_em')
            dr_state_t = dr_state[name].view(-1, 1)
            dr_state_t = torch.squeeze(embed(dr_state_t))
            V_dri.append(dr_state_t)

