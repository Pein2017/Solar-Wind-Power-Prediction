import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class CombinedEmbeddingDataset(Dataset):
    def __init__(
        self,
        root_path,
        train_path,
        test_path,
        size: List[int],
        flag: str,
        target: str = "power",
        scale_x_flag=True,
        scale_y_flag=True,
        timeenc: int = 1,
        freq: str = "15min",
        scaler_type="standard",
        irradiance_column="Solar Irradiance",
        train_val_ratio=[0.9, 0.1],
    ):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_val_ratio = train_val_ratio

        # Initialization
        assert flag in [
            "train",
            "test",
            "val",
        ], "flag must be 'train', 'test', or 'val'"

        # self.features = features
        self.target = target

        self.scale_x_flag = scale_x_flag
        self.scale_y_flag = scale_y_flag
        self.scaler_type = scaler_type

        self.timeenc = timeenc
        self.freq = freq

        self.flag = flag

        self.root_path = root_path
        self.train_path = train_path
        self.test_path = test_path

        self._load_and_process_data()

    def _load_and_process_data(self):
        # Load data
        df_train = pd.read_csv(os.path.join(self.root_path, self.train_path))
        df_test = pd.read_csv(os.path.join(self.root_path, self.test_path))

        # Combine train and test for consistent processing
        df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

        # Extract columns
        cols_data = df_combined.columns[
            1:
        ]  # Assume first column is date or non-feature
        df_data_x = df_combined[cols_data[:-1]]
        df_data_y = df_combined[cols_data[-1:]]

        # Split indices for training and validation
        total_records = len(df_train)
        train_ratio, val_ratio = self.train_val_ratio
        train_split = int(train_ratio * total_records)
        val_split = int(val_ratio * total_records)

        # Initialize scalers
        if self.scale_x_flag:
            self.scaler_x = StandardScaler()
        if self.scale_y_flag:
            self.scaler_y = (
                StandardScaler() if self.scaler_type == "standard" else MinMaxScaler()
            )

        # Fit scalers on the training data
        if self.scale_x_flag:
            self.scaler_x.fit(df_data_x.iloc[:train_split])
            data_x_scaled = self.scaler_x.transform(df_data_x)
        else:
            data_x_scaled = df_data_x.values

        if self.scale_y_flag:
            self.scaler_y.fit(df_data_y.iloc[:train_split])
            data_y_scaled = self.scaler_y.transform(df_data_y)
        else:
            data_y_scaled = df_data_y.values

        # Process time features
        df_stamp = pd.to_datetime(df_combined["date"])
        time_index = (df_stamp.dt.hour * 4 + df_stamp.dt.minute // 15) % 96
        data_stamp = time_index.values[:, None]  # Adding an extra dimension

        # Split the scaled data into train, validation, and test sets
        self.data_x_train = data_x_scaled[:train_split]
        self.data_y_train = data_y_scaled[:train_split]
        self.data_stamp_train = data_stamp[:train_split]

        self.data_x_val = data_x_scaled[train_split : train_split + val_split]
        self.data_y_val = data_y_scaled[train_split : train_split + val_split]
        self.data_stamp_val = data_stamp[train_split : train_split + val_split]

        self.data_x_test = data_x_scaled[total_records:]
        self.data_y_test = data_y_scaled[total_records:]
        self.data_stamp_test = data_stamp[total_records:]

        # Set the data for the current flag
        if self.flag == "train":
            self.data_x = self.data_x_train
            self.data_y = self.data_y_train
            self.data_stamp = self.data_stamp_train
        elif self.flag == "val":
            self.data_x = self.data_x_val
            self.data_y = self.data_y_val
            self.data_stamp = self.data_stamp_val
        else:  # test
            self.data_x = self.data_x_test
            self.data_y = self.data_y_test
            self.data_stamp = self.data_stamp_test

        print(
            f"flag = {self.flag}, data_x.shape = {self.data_x.shape}, data_y.shape = {self.data_y.shape}, data_stamp.shape = {self.data_stamp.shape}"
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - 1
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x)  # , dtype=torch.float16
        seq_y = torch.tensor(seq_y)
        seq_x_mark = torch.tensor(seq_x_mark)
        seq_y_mark = torch.tensor(seq_y_mark)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class FullDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_path,
        size: List[int],
        flag: str = "train",
        target: str = "power",
        scale_x_flag=True,
        scale_y_flag=True,
        timeenc: int = 1,
        freq: str = "15min",
        seasonal_patterns=None,
        scaler=None,
        scaler_type="standard",
        irradiance_column="Solar Irradiance",
        train_val_ratio=[0.7, 0.1],
        convert_to_beijing_time=False,
    ):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_val_ratio = train_val_ratio

        # Initialization
        assert flag in [
            "train",
            "test",
            "val",
        ], "flag must be 'train', 'test', or 'val'"

        type_map = {"train": 0, "val": 1, "test": 2}
        self.train_type = type_map[flag]

        # self.features = features
        self.target = target
        self.scale_x_flag = scale_x_flag
        self.scale_y_flag = scale_y_flag

        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        self.scaler_type = scaler_type
        self.convert_to_beijing_time = convert_to_beijing_time

        #! hardcore

        self.minmax_features = [
            "siconc",
            "crr",
            "csfr",
            "fzra",
            "hcc",
            "tciw",
            "ilspf",
            "litoti",
            "lsrr",
            "lssfr",
            "ptype",
            "sd",
            "power",
        ]

        # self.irradiance_column = irradiance_column

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        total_records = len(df_raw)
        if self.convert_to_beijing_time:
            # Convert the 'date' column from UTC to Beijing Time
            df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
            df_raw["date"] = df_raw["date"].dt.tz_convert("Asia/Shanghai")

            # If you want to remove the timezone information after conversion
            df_raw["date"] = df_raw["date"].dt.tz_localize(None)

        train_ratio, val_ratio = self.train_val_ratio
        train_split = int(float(train_ratio) * total_records)  # 70% for training
        val_split = int(float(val_ratio) * total_records)  # 10% for validation
        print(
            f"total records = {total_records}, train_split = {train_split}, val_split = {val_split}"
        )

        border1s = [0, train_split, train_split + val_split]
        border2s = [train_split, train_split + val_split, total_records]

        cols_data = df_raw.columns[1:]  # Assume first column is date or non-feature
        df_data = df_raw[cols_data]

        border1 = border1s[self.train_type]
        border2 = border2s[self.train_type]

        # Split the data for training only for fitting the scaler
        train_data_x = df_data.iloc[border1s[0] : border2s[0], :-1]
        train_data_y = df_data.iloc[border1s[0] : border2s[0], -1:]

        if self.scale_x_flag:
            self.scaler_standard = StandardScaler()
            self.scaler_minmax = MinMaxScaler()

            # 确保 minmax_features 存在于数据集中
            available_minmax_features = [
                feat for feat in self.minmax_features if feat in train_data_x.columns
            ]

            print(f"Found available minmax features: {available_minmax_features}")

            # Fit the scalers on the training data excluding special tokens
            if available_minmax_features:
                self.scaler_minmax.fit(train_data_x[available_minmax_features])
                df_data_scaled_minmax = self.scaler_minmax.transform(
                    df_data[available_minmax_features]
                )
                df_data_scaled_minmax = pd.DataFrame(
                    df_data_scaled_minmax, columns=available_minmax_features
                )
            else:
                df_data_scaled_minmax = pd.DataFrame()

            standard_features = train_data_x.drop(
                columns=available_minmax_features
            ).columns
            self.scaler_standard.fit(train_data_x[standard_features])
            df_data_scaled_standard = self.scaler_standard.transform(
                df_data[standard_features]
            )
            df_data_scaled_standard = pd.DataFrame(
                df_data_scaled_standard, columns=standard_features
            )

            # 合并标准化的特征
            if not df_data_scaled_minmax.empty:
                df_data_scaled = pd.concat(
                    [df_data_scaled_standard, df_data_scaled_minmax], axis=1
                )
            else:
                df_data_scaled = df_data_scaled_standard

            df_data_scaled = df_data_scaled[train_data_x.columns]  # 保持原始列顺序

        else:
            df_data_scaled = df_data.iloc[:, :-1]

        if self.scale_y_flag:
            if self.scaler_type == "standard":
                self.scaler_y = StandardScaler()
            else:
                self.scaler_y = MinMaxScaler()

            # Extract indices of NaN values in train_data_y
            nan_indices_train_y = train_data_y.isna()

            # Fit scaler on the training data excluding NaN values
            self.scaler_y.fit(train_data_y[~nan_indices_train_y].dropna())

            df_data_y_scaled = self.scaler_y.transform(df_data.iloc[:, -1:])

        else:
            df_data_y_scaled = df_data.iloc[:, -1:]

        self.data_x = df_data_scaled
        self.data_y = df_data_y_scaled

        # Process time features
        df_stamp = df_raw[["date"]][border1:border2]

        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        # 生成时间特征
        x_mark = self._generate_time_features(df_stamp)

        self.data_stamp = x_mark

        df_stamp_ori = df_raw[["date"]][border1:border2].copy()
        df_stamp_ori["date"] = pd.to_datetime(df_stamp_ori["date"])

        data_combined = np.concatenate(
            (self.data_x.values, self.data_y), axis=1
        )  # new_features.values,
        # print(f"data_combined.shape = {data_combined.shape}")

        self.data_x = data_combined[border1:border2, :-1]
        self.data_y = data_combined[border1:border2, -1:]

        print(f"data_x.shape = {self.data_x.shape}")
        print(f"data_stamp.shape = {x_mark.shape}")
        print(f"data_y.shape = {self.data_y.shape}")
        print(f"total columns = {len(df_raw.columns)}")
        assert len(df_raw.columns) == int(self.data_x.shape[1]) + 2, "数据处理有误"
        self.total_column = len(df_raw.columns)

        self.time_stamp = df_raw[["date"]][border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - 1
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def _generate_time_features(self, df_stamp, include_extra=False):
        # 提取基本时间特征
        df_stamp["month"] = df_stamp["date"].dt.month
        df_stamp["day"] = df_stamp["date"].dt.day
        df_stamp["weekday"] = df_stamp["date"].dt.weekday
        df_stamp["hour"] = df_stamp["date"].dt.hour
        df_stamp["minute"] = df_stamp["date"].dt.minute // 15  # 转换成每小时的4个间隔

        if include_extra:
            # Apply sine and cosine transformations
            df_stamp["hour_sin"] = np.sin(2 * np.pi * df_stamp["hour"] / 24)
            df_stamp["hour_cos"] = np.cos(2 * np.pi * df_stamp["hour"] / 24)
            df_stamp["day_sin"] = np.sin(2 * np.pi * df_stamp["day"] / 31)
            df_stamp["day_cos"] = np.cos(2 * np.pi * df_stamp["day"] / 31)
            df_stamp["dayofweek_sin"] = np.sin(2 * np.pi * df_stamp["weekday"] / 7)
            df_stamp["dayofweek_cos"] = np.cos(2 * np.pi * df_stamp["weekday"] / 7)
            df_stamp["month_sin"] = np.sin(2 * np.pi * df_stamp["month"] / 12)
            df_stamp["month_cos"] = np.cos(2 * np.pi * df_stamp["month"] / 12)

            # Combine all features into x_mark
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
                    "dayofweek_sin",
                    "dayofweek_cos",
                    "month_sin",
                    "month_cos",
                ]
            ].values
        else:
            # Combine basic features into x_mark
            x_mark = df_stamp[["month", "day", "weekday", "hour", "minute"]].values

        return x_mark


class SolarMaskDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_path,
        size: List[int],
        flag: str = "train",
        target: str = "power",
        scale_x_flag=True,
        scale_y_flag=False,
        timeenc: int = 1,
        freq: str = "15min",
        seasonal_patterns=None,
        scaler=None,
        scaler_type="minmax",
        irradiance_column="Solar Irradiance",
        train_val_ratio=[0.7, 0.1],
        convert_to_beijing_time=True,
    ):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.train_val_ratio = train_val_ratio

        # Initialization
        assert flag in [
            "train",
            "test",
            "val",
        ], "flag must be 'train', 'test', or 'val'"

        type_map = {"train": 0, "val": 1, "test": 2}
        self.train_type = type_map[flag]

        # self.features = features
        self.target = target
        self.scale_x_flag = scale_x_flag
        self.scale_y_flag = scale_y_flag

        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        self.scaler_type = scaler_type

        self.irradiance_column = irradiance_column
        self.convert_to_beijing_time = convert_to_beijing_time

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.total_column = len(df_raw.columns)

        total_records = len(df_raw)

        if self.convert_to_beijing_time:
            # Convert the 'date' column from UTC to Beijing Time
            df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
            df_raw["date"] = df_raw["date"].dt.tz_convert("Asia/Shanghai")

            # If you want to remove the timezone information after conversion
            df_raw["date"] = df_raw["date"].dt.tz_localize(None)

        train_ratio, val_ratio = self.train_val_ratio
        train_split = int(float(train_ratio) * total_records)  # 70% for training
        val_split = int(float(val_ratio) * total_records)  # 10% for validation
        print(
            f"total records = {total_records}, train_split = {train_split}, val_split = {val_split}"
        )

        border1s = [0, train_split, train_split + val_split]
        border2s = [train_split, train_split + val_split, total_records]

        cols_data = df_raw.columns[1:]  # Assume first column is date or non-feature
        df_data = df_raw[cols_data]

        border1 = border1s[self.train_type]
        border2 = border2s[self.train_type]

        # Split the data for training only for fitting the scaler
        train_data_x = df_data.iloc[border1s[0] : border2s[0], :-1]
        train_data_y = df_data.iloc[border1s[0] : border2s[0], -1:]

        if self.scale_x_flag:
            if self.scaler_type == "standard":
                self.scaler_x = StandardScaler()
            else:
                self.scaler_x = MinMaxScaler()

            # Fit the scaler on the training data excluding special tokens
            self.scaler_x.fit(train_data_x[~train_data_x.isna().any(axis=1)])

            df_data_scaled = self.scaler_x.transform(df_data.iloc[:, :-1])

            # Convert back to DataFrame to restore the original markers
            df_data_scaled = pd.DataFrame(df_data_scaled, columns=df_data.columns[:-1])

        else:
            df_data_scaled = df_data.iloc[:, :-1]

        if self.scale_y_flag:
            if self.scaler_type == "standard":
                self.scaler_y = StandardScaler()
            else:
                self.scaler_y = MinMaxScaler()

            # Extract indices of NaN values in train_data_y
            nan_indices_train_y = train_data_y.isna()

            # Fit scaler on the training data excluding NaN values
            self.scaler_y.fit(train_data_y[~nan_indices_train_y].dropna())

            df_data_y_scaled = self.scaler_y.transform(df_data.iloc[:, -1:])

        else:
            df_data_y_scaled = df_data.iloc[:, -1:]

        self.data_x = df_data_scaled
        self.data_y = df_data_y_scaled

        # Process time features
        df_stamp = df_raw[["date"]][border1:border2]

        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        # 生成时间特征
        x_mark = self._generate_time_features(df_stamp)

        self.data_stamp = x_mark

        df_stamp_ori = df_raw[["date"]][border1:border2].copy()
        df_stamp_ori["date"] = pd.to_datetime(df_stamp_ori["date"])

        data_combined = np.concatenate(
            (self.data_x.values, self.data_y), axis=1
        )  # new_features.values,
        # print(f"data_combined.shape = {data_combined.shape}")

        self.data_x = data_combined[border1:border2, :-1]
        self.data_y = data_combined[border1:border2, -1:]

        print(f"data_x.shape = {self.data_x.shape}")
        print(f"data_stamp.shape = {x_mark.shape}")
        print(f"data_y.shape = {self.data_y.shape}")

        self.time_stamp = df_raw[["date"]][border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - 1
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def _generate_time_features(self, df_stamp, include_extra=False):
        # 提取基本时间特征
        df_stamp["month"] = df_stamp["date"].dt.month
        df_stamp["day"] = df_stamp["date"].dt.day
        df_stamp["weekday"] = df_stamp["date"].dt.weekday
        df_stamp["hour"] = df_stamp["date"].dt.hour
        df_stamp["minute"] = df_stamp["date"].dt.minute // 15  # 转换成每小时的4个间隔

        if include_extra:
            # Apply sine and cosine transformations
            df_stamp["hour_sin"] = np.sin(2 * np.pi * df_stamp["hour"] / 24)
            df_stamp["hour_cos"] = np.cos(2 * np.pi * df_stamp["hour"] / 24)
            df_stamp["day_sin"] = np.sin(2 * np.pi * df_stamp["day"] / 31)
            df_stamp["day_cos"] = np.cos(2 * np.pi * df_stamp["day"] / 31)
            df_stamp["dayofweek_sin"] = np.sin(2 * np.pi * df_stamp["weekday"] / 7)
            df_stamp["dayofweek_cos"] = np.cos(2 * np.pi * df_stamp["weekday"] / 7)
            df_stamp["month_sin"] = np.sin(2 * np.pi * df_stamp["month"] / 12)
            df_stamp["month_cos"] = np.cos(2 * np.pi * df_stamp["month"] / 12)

            # Combine all features into x_mark
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
                    "dayofweek_sin",
                    "dayofweek_cos",
                    "month_sin",
                    "month_cos",
                ]
            ].values
        else:
            # Combine basic features into x_mark
            x_mark = df_stamp[["month", "day", "weekday", "hour", "minute"]].values

        return x_mark
