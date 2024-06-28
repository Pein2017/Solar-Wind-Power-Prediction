import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     mape = np.abs((pred - true) / true)
#     mape = np.where(mape > 5, 0, mape)
#     return np.mean(mape)


# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))


def MAE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    return np.mean(np.abs(filtered_pred - filtered_true))


def MAPE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    mape = np.abs((filtered_pred - filtered_true) / filtered_true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true, threshold=1.0):
    mask = true >= threshold
    filtered_pred = pred[mask]
    filtered_true = true[mask]
    return np.mean(np.square((filtered_pred - filtered_true) / filtered_true))


def metric(pred, true):
    mask = ~np.isnan(true)

    nan_count = np.isnan(true).sum()
    if nan_count > 0:
        print(f"Number of NaN values in true: {nan_count} during metric calculation")

    pred = pred[mask]
    true = true[mask]

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
