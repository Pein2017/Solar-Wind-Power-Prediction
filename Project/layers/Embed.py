import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, combine_type="add"):
        super(TemporalEmbedding, self).__init__()

        self.combine_type = combine_type

        # Define linear layers for mins_position, mins_position_sin, and mins_position_cos
        self.linear_pos = nn.Linear(1, d_model)
        self.linear_sin = nn.Linear(1, d_model)
        self.linear_cos = nn.Linear(1, d_model)

        # If combining by concatenation, define a linear layer to project the concatenated output to d_model
        if self.combine_type == "concat":
            self.combine_linear = nn.Linear(3 * d_model, d_model)

        self.dropout = nn.Dropout(0.1)  # Dropout rate (optional)

    def forward(self, x):
        # x: [B, L, 3] - assuming x contains mins_position, sine, and cosine transformations as the last three features
        mins_position = x[:, :, 0].unsqueeze(
            -1
        )  # Extract mins_position and add an extra dimension
        mins_position_sin = x[:, :, 1].unsqueeze(
            -1
        )  # Extract sine transformation and add an extra dimension
        mins_position_cos = x[:, :, 2].unsqueeze(
            -1
        )  # Extract cosine transformation and add an extra dimension

        # Apply linear layers to the transformations
        pos_embedding = self.linear_pos(mins_position)
        sin_embedding = self.linear_sin(mins_position_sin)
        cos_embedding = self.linear_cos(mins_position_cos)

        # Combine embeddings
        if self.combine_type == "add":
            combined_embedding = pos_embedding + sin_embedding + cos_embedding
        elif self.combine_type == "concat":
            combined_embedding = torch.cat(
                [pos_embedding, sin_embedding, cos_embedding], dim=-1
            )
            combined_embedding = self.combine_linear(combined_embedding)

        # Apply dropout (optional)
        combined_embedding = self.dropout(combined_embedding)

        return combined_embedding


class CombinedTemporalEmbedding(nn.Module):
    def __init__(self, d_model, combine_type="add"):
        super(CombinedTemporalEmbedding, self).__init__()
        self.d_model = d_model
        self.combine_type = combine_type

        self.linear_pos = nn.Linear(1, d_model)
        self.linear_pos_sin = nn.Linear(1, d_model)
        self.linear_pos_cos = nn.Linear(1, d_model)

        self.linear_hour = nn.Linear(1, d_model)
        self.linear_hour_sin = nn.Linear(1, d_model)
        self.linear_hour_cos = nn.Linear(1, d_model)

        if self.combine_type == "concat":
            self.combine_linear = nn.Linear(6 * d_model, d_model)

        self.dropout = nn.Dropout(0.1)  # Dropout rate (optional)

    def forward(self, x):
        # x: [B, L, 6] - containing mins_position, mins_position_sin, mins_position_cos, hour, hour_sin, hour_cos
        x = x.squeeze(
            2
        )  # Now x.shape should be (batch*num_feature, seq_len, num_time_feature)
        # Extract features
        mins_position = x[:, :, 0].unsqueeze(-1)
        mins_position_sin = x[:, :, 1].unsqueeze(-1)
        mins_position_cos = x[:, :, 2].unsqueeze(-1)
        hour = x[:, :, 3].unsqueeze(-1)
        hour_sin = x[:, :, 4].unsqueeze(-1)
        hour_cos = x[:, :, 5].unsqueeze(-1)

        # Apply linear layers to the transformations
        pos_embedding = self.linear_pos(mins_position)
        pos_sin_embedding = self.linear_pos_sin(mins_position_sin)
        pos_cos_embedding = self.linear_pos_cos(mins_position_cos)

        hour_embedding = self.linear_hour(hour)
        hour_sin_embedding = self.linear_hour_sin(hour_sin)
        hour_cos_embedding = self.linear_hour_cos(hour_cos)

        # Combine embeddings
        if self.combine_type == "concat":
            combined_embedding = torch.cat(
                [
                    pos_embedding,
                    pos_sin_embedding,
                    pos_cos_embedding,
                    hour_embedding,
                    hour_sin_embedding,
                    hour_cos_embedding,
                ],
                dim=-1,
            )
            combined_embedding = self.combine_linear(combined_embedding)
        else:
            combined_embedding = (
                pos_embedding
                + pos_sin_embedding
                + pos_cos_embedding
                + hour_embedding
                + hour_sin_embedding
                + hour_cos_embedding
            )

        # Apply dropout (optional)
        combined_embedding = self.dropout(combined_embedding)

        return combined_embedding


class FinalEmbedding(nn.Module):
    def __init__(self, c_in, d_model, combine_type="add"):
        super(FinalEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = CombinedTemporalEmbedding(d_model, combine_type)
        self.combine_type = combine_type

        if self.combine_type == "concat":
            self.final_linear = nn.Linear(2 * d_model, d_model)

    def forward(self, x, time_index):
        # x: [B, L, c_in]
        token_emb = self.token_embedding(x)  # Token embedding

        # Prepare temporal input
        mins_position = time_index.unsqueeze(
            -1
        ).float()  # Convert to float and add extra dimension
        mins_position_sin = torch.sin(2 * np.pi * mins_position / 96)
        mins_position_cos = torch.cos(2 * np.pi * mins_position / 96)
        hour = (
            (time_index // 4).unsqueeze(-1).float()
        )  # Convert to float and add extra dimension
        hour_sin = torch.sin(2 * np.pi * hour / 24)
        hour_cos = torch.cos(2 * np.pi * hour / 24)

        temporal_input = torch.cat(
            [
                mins_position,
                mins_position_sin,
                mins_position_cos,
                hour,
                hour_sin,
                hour_cos,
            ],
            dim=-1,
        )

        temporal_emb = self.temporal_embedding(temporal_input)  # Temporal embedding

        # Combine embeddings
        if self.combine_type == "concat":
            combined_emb = torch.cat([token_emb, temporal_emb], dim=-1)
            combined_emb = self.final_linear(combined_emb)
        else:
            combined_emb = token_emb + temporal_emb

        return combined_emb


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None and x_mark is not None:
            return self.temporal_embedding(x_mark)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# 生成时间序列数据
def generate_time_series(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq="15T")
    df_stamp = pd.DataFrame({"date": dates})
    df_stamp["month"] = df_stamp["date"].dt.month
    df_stamp["day"] = df_stamp["date"].dt.day
    df_stamp["weekday"] = df_stamp["date"].dt.weekday
    df_stamp["hour"] = df_stamp["date"].dt.hour
    df_stamp["minute"] = df_stamp["date"].dt.minute
    df_stamp["hour_sin"] = np.sin(2 * np.pi * df_stamp["hour"] / 24)
    df_stamp["hour_cos"] = np.cos(2 * np.pi * df_stamp["hour"] / 24)
    df_stamp["day_sin"] = np.sin(2 * np.pi * df_stamp["day"] / 31)
    df_stamp["day_cos"] = np.cos(2 * np.pi * df_stamp["day"] / 31)
    df_stamp["weekday_sin"] = np.sin(2 * np.pi * df_stamp["weekday"] / 7)
    df_stamp["weekday_cos"] = np.cos(2 * np.pi * df_stamp["weekday"] / 7)
    df_stamp["month_sin"] = np.sin(2 * np.pi * df_stamp["month"] / 12)
    df_stamp["month_cos"] = np.cos(2 * np.pi * df_stamp["month"] / 12)
    x_mark = df_stamp[
        [
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]
    ].values
    return x_mark


def visualize_temporal_embedding(model, x_mark, filename="temporal_embedding.png"):
    with torch.no_grad():
        embeddings = model(x_mark)
    embeddings = embeddings.squeeze().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(embeddings)
    plt.title("Temporal Embeddings Visualization")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.legend(["Hour", "Weekday", "Day", "Month", "Minute"])
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def visualize_solar_embedding(model, time_index, filename="solar_embedding.png"):
    with torch.no_grad():
        embeddings = model(time_index)
    embeddings = embeddings.squeeze().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(embeddings)
    plt.title("Solar Irradiance Embeddings Visualization")
    plt.xlabel("Time Step")
    plt.ylabel("Embedding Value")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    batch_size, seq_len, c_in = 32, 10, 3  # Example dimensions
    d_model = 2  # Dimension of the model

    # Example input tensors
    x = torch.randn(batch_size, seq_len, c_in)  # Input features
    time_index = torch.randint(0, 96, (batch_size, seq_len))  # Time indices

    # Create and test the final embedding layer
    final_embedding_add = FinalEmbedding(c_in, d_model, combine_type="add")
    final_embedding_concat = FinalEmbedding(c_in, d_model, combine_type="concat")

    output_add = final_embedding_add(x, time_index)
    output_concat = final_embedding_concat(x, time_index)

    print(
        "Output shape with 'add':", output_add.shape
    )  # Should output torch.Size([32, 10, 64])
    print(
        "Output shape with 'concat':", output_concat.shape
    )  # Should output torch.Size([32, 10, 64])
