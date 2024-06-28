import torch  # noqa
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.in_features = config.enc_in
        self.out_features = config.c_out
        self.hidden_size = config.d_model
        self.pred_len = config.pred_len
        self.num_hidden_layers = config.e_layers

        self.flattened_size = self.seq_len * self.in_features

        layers = []
        layers.append(nn.Linear(self.flattened_size, self.hidden_size))
        layers.append(nn.ReLU())
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size, self.pred_len * self.out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x, *args):
        # Flatten the input from [batch_size, seq_len, in_features] to [batch_size, seq_len * in_features]
        x = x.view(x.size(0), -1)  # Flatten while preserving batch size
        out = self.network(x)
        # Reshape output to [batch_size, pred_len, out_features]
        out = out.view(x.size(0), self.pred_len, self.out_features)
        return out
