import torch
import torch.nn as nn
from utils import shape


class SequenceGRU(nn.Module):
    def __init__(self, input_size_is_num_features, hidden_size):
        super(SequenceGRU, self).__init__()
        self.input_size = (
            input_size_is_num_features  # For us it's 1 since each timestep is 1 number
        )
        self.hidden_size = hidden_size  # arbitrary
        self.gru1 = nn.GRU(
            input_size_is_num_features, hidden_size, 1
        )  # one layer in GRU
        self.linear = nn.Linear(hidden_size, 1)  # Infer just 1 future step

    def forward(self, input):
        # input of shape (in_a_batch, seq_len_is_timesteps_per_sequence, num_features_input_size)
        input = shape(input)
        batch_size = input.size(1)

        # h0 = torch.randn(num_layers, in_a_batch, hidden_size_arbitrary)
        h_t = input.new_zeros(
            1, batch_size, self.hidden_size, dtype=torch.double
        )  # 1 is num of layers

        gru_out, h_t = self.gru1(input, h_t)

        # Only take the output from the final timetep - gru_out[-1]
        y_pred = self.linear(gru_out[-1].view(batch_size, -1))
        return y_pred.view(-1)
