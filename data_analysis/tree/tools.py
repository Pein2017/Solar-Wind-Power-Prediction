import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.learning_rate
            if epoch < 3
            else args.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "PEMS":
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    data = pd.DataFrame({"true": true, "preds": preds})
    data.to_csv(name, index=False, sep=",")


def visual(true, preds, name_base="./pic/test"):
    """
    Results visualization
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(name_base), exist_ok=True)

    # Plot together
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(
        true, label="GroundTruth", color="blue", linewidth=1, marker="o", markersize=3
    )

    plt.plot(
        preds,
        label="Prediction",
        color="red",
        linestyle="--",
        linewidth=1,
        marker="s",
        markersize=3,
    )
    plt.legend()

    # Calculate residuals
    residuals = np.array(true) - np.array(preds)
    plt.subplot(2, 1, 2)
    plt.plot(
        residuals,
        label="Residual",
        color="green",
        linestyle="-",
        linewidth=1,
        marker="x",
        markersize=3,
    )
    plt.title("Residual")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name_base}_together.pdf", bbox_inches="tight")
    print(f"Figure saved to {name_base}_together.pdf")
    # Do not close the figure
    plt.show()

    # Plot separately
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plt.plot(
        true, label="GroundTruth", color="blue", linewidth=1, marker="o", markersize=3
    )
    plt.title("GroundTruth")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(
        preds,
        label="Prediction",
        color="red",
        linestyle="--",
        linewidth=1,
        marker="s",
        markersize=3,
    )
    plt.title("Prediction")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(
        residuals,
        label="Residual",
        color="green",
        linestyle="-",
        linewidth=1,
        marker="x",
        markersize=3,
    )
    plt.title("Residual")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{name_base}_separate.pdf", bbox_inches="tight")
    print(f"Figure saved to {name_base}_separate.pdf")
    # Do not close the figure
    plt.show()


def visual_weights(weights, name="./pic/test.pdf"):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap="YlGnBu")
    fig.colorbar(im, pad=0.03, location="top")
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def accuracy_metric_loss(pred, true, cap=200.0):
    """
    Compute the custom accuracy metric.

    Parameters:
    - pred: numpy array of predictions
    - true: numpy array of true values
    - cap: float, threshold for computing S_i

    Returns:
    - accuracy_metric: float, the computed accuracy metric
    """

    # Ensure both arrays are of the same shape
    assert (
        pred.shape == true.shape
    ), "Shape mismatch between prediction and ground truth arrays"

    # Mask out NaN values
    mask = ~np.isnan(true)
    pred = pred[mask]
    true = true[mask]

    # If no valid values, return NaN
    if true.size == 0:
        print("Warning: No valid true values.")
        return np.nan

    # Compute S_i based on the provided conditions
    S_i = np.where(true >= 0.2 * cap, true, 0.2 * cap)

    # Compute the custom accuracy measure
    diff = (true - pred) / S_i
    squared_diff = diff**2
    mean_squared_diff = np.mean(squared_diff)
    accuracy_metric = (1 - np.sqrt(mean_squared_diff)) * 100

    return accuracy_metric
