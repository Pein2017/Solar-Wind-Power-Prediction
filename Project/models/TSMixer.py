import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos


class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        # Temporal component
        self.temporal = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Dropout(configs.dropout),
        )

        # Channel component
        self.channel = nn.Sequential(
            nn.Linear(configs.seq_len, configs.seq_len),
            nn.ReLU(),
            nn.Linear(configs.seq_len, configs.seq_len),
            nn.Dropout(configs.dropout),
        )

    def forward(self, x):
        # x: [B, L, D]

        # Temporal processing
        x = x + self.temporal(x)

        # Channel processing
        x = x + self.channel(x.transpose(1, 2)).transpose(1, 2)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.use_embedding = configs.use_embedding
        self.c_out = configs.c_out

        if self.use_embedding:
            self.data_embedding = DataEmbedding_wo_pos(
                c_in=configs.enc_in,
                d_model=configs.d_model,
                embed_type=configs.embed,
                freq=configs.freq,
                dropout=configs.dropout,
            )

        self.model = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])

        # Projection for regression output
        self.regression_projection = nn.Linear(configs.d_model, self.c_out)

        # Additional classification head for zero vs non-zero prediction
        self.classification_head = nn.Linear(configs.d_model, self.c_out)

        # Final projection to pred_len
        self.final_projection = nn.Linear(configs.seq_len, self.pred_len)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.use_embedding:
            x_enc = self.data_embedding(x_enc, x_mark_enc)

        # Pass through ResBlock layers
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)

        # Apply final projection layer for regression output
        regression_out = self.regression_projection(x_enc)

        # Apply classification head for zero vs non-zero prediction
        classification_out = torch.sigmoid(self.classification_head(x_enc))

        # Combine regression and classification outputs
        combined_output = regression_out * classification_out

        # Project to prediction length
        combined_output = self.final_projection(
            combined_output.transpose(1, 2)
        ).transpose(1, 2)

        return combined_output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        else:
            raise ValueError("Only forecast tasks implemented yet")


def zero_inflated_loss(y_true, regression_out, classification_out):
    mask = (y_true > 0).float()
    regression_loss = F.mse_loss(regression_out * mask, y_true * mask)
    classification_loss = F.binary_cross_entropy(
        classification_out, (y_true > 0).float()
    )
    return regression_loss + classification_loss
