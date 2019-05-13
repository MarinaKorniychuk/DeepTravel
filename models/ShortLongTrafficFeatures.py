import torch
import torch.nn as nn


class ShortTermLSTM(nn.Module):

    def __init__(self, ):
        super(ShortTermLSTM, self).__init__()

        self.rnn_0 = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.rnn_1 = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.rnn_2 = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

    def forward(self, short_ttf):

        V_short = []

        for cell in short_ttf:

            input_0, input_1, input_2 = cell[0], cell[1], cell[2]
            input_0 = torch.unsqueeze(input_0, dim=0)
            input_1 = torch.unsqueeze(input_1, dim=0)
            input_2 = torch.unsqueeze(input_2, dim=0)

            outputs_0, (h_n, c_n) = self.rnn_0(input_0)
            outputs_1, (h_n, c_n) = self.rnn_1(input_1)
            outputs_2, (h_n, c_n) = self.rnn_2(input_2)

            hiddens_v = torch.squeeze(torch.cat([outputs_0[:, -1], outputs_1[:, -1], outputs_2[:, -1]], dim=1), dim=0)

            V_short.append(hiddens_v)

        return V_short


class LongTermLSTM(nn.Module):

    def __init__(self, ):
        super(LongTermLSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

    def forward(self, long_ttf):

        V_long = []

        for cell in long_ttf:
            input = torch.unsqueeze(cell, dim=0)
            outputs, (h_n, c_n) = self.rnn(input)

            V_long.append(torch.squeeze(outputs[:, -1], dim=0))

        return V_long
