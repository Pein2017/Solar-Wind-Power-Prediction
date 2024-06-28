import argparse
import random

import numpy as np
import torch
from exp import Exp

fix_seed = 17
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="TimesNet")

# self-customized
parser.add_argument(
    "--use_embedding",
    type=int,
    required=False,
    default=1,
    help="whether to use combined embedding",
)

parser.add_argument(
    "--scaler_type", type=str, required=False, default="standard", help="scaler type"
)


# basic config
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
)

parser.add_argument("--is_training", type=int, required=True, default=1, help="status")


parser.add_argument(
    "--model_id", type=str, required=True, default="test", help="model id"
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="Autoformer",
    help="model name, options: [Autoformer, Transformer, TimesNet]",
)

# data loader
parser.add_argument(
    "--data", type=str, required=True, default="ETTm1", help="dataset type"
)


parser.add_argument(
    "--root_path", type=str, default="./data/ETT/", help="root path of the data file"
)
parser.add_argument(
    "--scale_x_flag", type=int, default=1, help="whether to scale inpu features X"
)
parser.add_argument(
    "--scale_y_flag", type=int, default=1, help="whether to scale target features Y"
)
parser.add_argument("--train_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument("--test_path", type=str, default="ETTh1.csv", help="data file")

parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)


# forecasting task
parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)
parser.add_argument("--inverse", type=bool, help="inverse output data", default=True)

# model define
parser.add_argument(
    "--last_hidden_dim", type=int, default=16, help="last hidden dim of FFN"
)

parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
parser.add_argument("--enc_in", type=int, default=6, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=6, help="decoder input size")
parser.add_argument("--c_out", type=int, default=1, help="output size")
parser.add_argument("--d_model", type=int, default=16, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--d_ff", type=int, default=32, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="learned",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in ecoder",
)
parser.add_argument(
    "--channel_independence",
    type=int,
    default=1,
    help="0: channel dependence 1: channel independence for FreTS model",
)
parser.add_argument(
    "--decomp_method",
    type=str,
    default="moving_avg",
    help="method of series decompsition, only support moving_avg or dft_decomp",
)
parser.add_argument(
    "--use_norm", type=int, default=1, help="whether to use normalize; True 1 False 0"
)
parser.add_argument(
    "--down_sampling_layers", type=int, default=0, help="num of down sampling layers"
)
parser.add_argument(
    "--down_sampling_window", type=int, default=1, help="down sampling window size"
)
parser.add_argument(
    "--down_sampling_method",
    type=str,
    default="avg",
    help="down sampling method, only support avg, max, conv",
)
parser.add_argument(
    "--use_future_temporal_feature",
    type=int,
    default=0,
    help="whether to use future_temporal_feature; True 1 False 0",
)


# optimization
parser.add_argument(
    "--num_workers", type=int, default=0, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="optimizer learning rate"
)

parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="TST", help="adjust learning rate")
parser.add_argument("--pct_start", type=float, default=0.2, help="pct_start")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)
parser.add_argument("--comment", type=str, default="none", help="com")

# NPU
parser.add_argument("--use_npu", type=bool, default=False, help="use npu")
parser.add_argument("--npu", type=int, default=0, help="npu device id")
parser.add_argument(
    "--use_multi_npu", action="store_true", help="use multiple npus", default=False
)
parser.add_argument(
    "--devices", type=str, default="0", help="device ids of multiple npus"
)

# de-stationary projector params
parser.add_argument(
    "--p_hidden_dims",
    type=int,
    nargs="+",
    default=[128, 128],
    help="hidden layer dimensions of projector (List)",
)
parser.add_argument(
    "--p_hidden_layers",
    type=int,
    default=2,
    help="number of hidden layers in projector",
)


try:
    args = parser.parse_args()
except Exception as e:
    print(f"Argument error: {e}")


# args = parser.parse_args()
if args.use_npu:
    import torch_npu  # noqa

    args.use_npu = True if torch.npu.is_available() else False
    torch.npu.set_device(args.npu)
else:
    args.use_npu = False

if args.use_npu and args.use_multi_npu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.npu = args.device_ids[0]


print("Args in experiment:")
print(args)


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = (
            f"{args.comment}_sl{args.seq_len}_lr{args.learning_rate}_feat{args.enc_in}"
        )
        exp = Exp(args)  # set experiments
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        torch.npu.empty_cache()
else:
    ii = 0
    setting = (
        "{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.comment,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )
    )

    exp = Exp(args)  # set experiments
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    exp.test(setting, test=1)
    torch.npu.empty_cache()
