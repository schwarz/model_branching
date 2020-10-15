import torch
import torch.nn as nn
from nn.utils import shape


# Credit: https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
class SequenceLSTM(nn.Module):
    def __init__(self, input_size_is_num_features, hidden_size):
        super(SequenceLSTM, self).__init__()
        self.input_size = (
            input_size_is_num_features  # For us it's 1 since each timestep is 1 number
        )
        self.hidden_size = hidden_size  # arbitrary
        self.lstm1 = nn.LSTM(
            input_size_is_num_features, hidden_size, 1
        )  # one layer in LSTM
        self.linear = nn.Linear(hidden_size, 1)  # Infer just 1 future step

    def forward(self, input):
        # input of shape (in_a_batch, seq_len_is_timesteps_per_sequence, num_features_input_size)
        input = shape(input)
        batch_size = input.size(1)

        # h0 = torch.randn(num_layers, in_a_batch, hidden_size_arbitrary)
        # use .new_zeros to ensure it's on the same device (cpu/gpu)
        h_t = input.new_zeros(
            1, batch_size, self.hidden_size, dtype=torch.double
        )  # 1 is num of layers
        c_t = input.new_zeros(1, batch_size, self.hidden_size, dtype=torch.double)

        lstm_out, (h_t, c_t) = self.lstm1(input, (h_t, c_t))

        # Only take the output from the final timetep - lstm_out[-1]
        y_pred = self.linear(lstm_out[-1].view(batch_size, -1))
        return y_pred.view(-1)
