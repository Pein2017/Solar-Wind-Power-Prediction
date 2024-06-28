from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_M4,
    Dataset_PEMS,
    Dataset_Solar,
    MSLSegLoader,
    PSMSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SolarMaskDataset,
    SWATSegLoader,
)
from torch.utils.data import DataLoader

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "m4": Dataset_M4,
    "PSM": PSMSegLoader,
    "MSL": MSLSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "SWAT": SWATSegLoader,
    "PEMS": Dataset_PEMS,
    "Solar": Dataset_Solar,
    # "SolarPower": SolarPowerDataset,
    "SolarMaskDataset": SolarMaskDataset,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 1 if args.embed == "timeF" else 0

    if flag == "test":
        shuffle_flag = False
        drop_last = False

        batch_size = args.batch_size
        freq = args.freq

    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 1:
        print(f"flag {flag}, batch size {batch_size}")
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            flag=flag,
            target=args.target,
            scale_x_flag=args.scale_x_flag,
            scale_y_flag=args.scale_y_flag,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
