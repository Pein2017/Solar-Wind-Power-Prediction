import sys

sys.path.insert(0, "/data/Pein/Pytorch/Wind-Solar-Prediction/TimeMixer")

from data_provider.data_factory import data_provider


class Args:
    root_path = "/data/Pein/Pytorch/Wind-Solar-Prediction/data"
    data_path = "solar_n_2.csv"
    data = "SolarMaskDataset"
    seq_len = 3
    label_len = 0
    pred_len = 1
    features = "MS"
    target = "POWER"
    scale = False
    timeenc = 0
    freq = "15min"
    num_workers = 2
    batch_size = 1
    embed = "timeF"
    task_name = "Solar"
    seasonal_patterns = None
    scale_x_flag = True
    scale_y_flag = True


args = Args()
train_set, train_loader = data_provider(
    args,
    flag="train",
)
val_set, val_loader = data_provider(
    args,
    flag="val",
)
test_set, test_loader = data_provider(
    args,
    flag="test",
)

print(len(train_loader), len(val_loader), len(test_loader))


start, end = 30, 60
for idx in range(start, end):
    first_sample = test_set[idx]
    print(f"time is {test_set.time_stamp.iloc[idx]}")

    # Unpack the sample
    seq_x, seq_y, seq_x_mark, seq_y_mark = first_sample
    print(seq_x[0])
    print(seq_y[0])


for a in test_loader:
    print(a)
    break
