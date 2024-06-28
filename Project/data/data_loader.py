from torch.utils.data import DataLoader

from .dataset import CombinedEmbeddingDataset, FullDataset, SolarMaskDataset

data_dict = {
    "SolarMaskDataset": SolarMaskDataset,
    "FullDataset": FullDataset,
}


def data_provider(args, flag):
    # Load and process data only once
    if not hasattr(data_provider, "data_train"):
        dataset = CombinedEmbeddingDataset(
            root_path=args.root_path,
            train_path=args.train_path,
            test_path=args.test_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag="train",  # Initial flag to load data
            target=args.target,
            scale_x_flag=args.scale_x_flag,
            scale_y_flag=args.scale_y_flag,
            timeenc=1 if args.embed == "timeF" else 0,
            freq=args.freq,
            scaler_type=args.scaler_type,
        )
        # Store the preprocessed data and scalers as class attributes
        data_provider.data_train = (
            dataset.data_x_train,
            dataset.data_y_train,
            dataset.data_stamp_train,
        )
        data_provider.data_val = (
            dataset.data_x_val,
            dataset.data_y_val,
            dataset.data_stamp_val,
        )
        data_provider.data_test = (
            dataset.data_x_test,
            dataset.data_y_test,
            dataset.data_stamp_test,
        )
        data_provider.scaler_x = dataset.scaler_x
        data_provider.scaler_y = dataset.scaler_y

    if flag == "train":
        data_x, data_y, data_stamp = data_provider.data_train
    elif flag == "val":
        data_x, data_y, data_stamp = data_provider.data_val
    else:  # test
        data_x, data_y, data_stamp = data_provider.data_test

    shuffle_flag = flag == "train"
    drop_last = flag == "train"
    batch_size = args.batch_size

    print(f"flag {flag}, batch size {batch_size}")

    data_set = CombinedEmbeddingDataset(
        root_path=args.root_path,
        train_path=args.train_path,
        test_path=args.test_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        flag=flag,
        target=args.target,
        scale_x_flag=args.scale_x_flag,
        scale_y_flag=args.scale_y_flag,
        timeenc=1 if args.embed == "timeF" else 0,
        freq=args.freq,
        scaler_type=args.scaler_type,
    )

    # Overwrite the data with preprocessed data
    data_set.data_x = data_x
    data_set.data_y = data_y
    data_set.data_stamp = data_stamp

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_set, data_loader


# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 1 if args.embed == "timeF" else 0

#     if flag == "test":
#         shuffle_flag = False
#         drop_last = False

#         batch_size = args.batch_size
#         freq = args.freq

#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size  # bsz for train and valid
#         freq = args.freq

#     print(f"flag {flag}, batch size {batch_size}")
#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         flag=flag,
#         target=args.target,
#         scale_x_flag=args.scale_x_flag,
#         scale_y_flag=args.scale_y_flag,
#         timeenc=timeenc,
#         freq=freq,
#         seasonal_patterns=args.seasonal_patterns,
#     )
#     print(flag, len(data_set))
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last,
#     )

#     return data_set, data_loader
