import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from data.data_loader import data_provider
from models import SimpleMLP, TimeMixer, TSMixer
from torch import optim
from torch.optim import lr_scheduler
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


class Exp:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimeMixer": TimeMixer,
            "SimpleMLP": SimpleMLP,
            "TSMixer": TSMixer,
        }
        self.device = self._acquire_device()

    def _acquire_device(self):
        if self.args.use_npu:
            import torch_npu  # noqa

            os.environ["NPU_VISIBLE_DEVICES"] = (
                str(self.args.npu) if not self.args.use_multi_npu else self.args.devices
            )
            device = torch.device("npu:{}".format(self.args.npu))
            if self.args.use_multi_npu:
                print("Use NPU: npu{}".format(self.args.device_ids))
            else:
                print("Use NPU: npu:{}".format(self.args.npu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_npu and self.args.use_npu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        # Get the number of features from the dataset
        num_features = data_set.data_x.shape[1]
        self.args.enc_in, self.args.dec_in = num_features, num_features
        print(
            f"self.args.enc_in is {self.args.enc_in}, self.args.dec_in is {self.args.dec_in}"
        )

        self.model = self._build_model().to(self.device)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == "PEMS":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch
                )
                dec_inp = None

                outputs = self._forward_model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )

                mask = batch_y == 0

                # Apply the mask to outputs: set the corresponding positions in outputs to 0
                outputs = torch.where(mask, torch.zeros_like(outputs), outputs)

                loss = self._compute_loss(outputs, batch_y, criterion)

                if not np.isnan(loss.item()):
                    total_loss.append(loss.item())
                else:
                    print(f"loss is inf at iter {i + 1}")

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, print_batch_interval: int = 100):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch
                )
                dec_inp = None

                outputs = self._forward_model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )

                # # Create a mask for batch_y where batch_y == 0
                # mask = batch_y == 0

                # # Apply the mask to outputs: set the corresponding positions in outputs to 0
                # outputs = torch.where(mask, torch.zeros_like(outputs), outputs)

                loss = self._compute_loss(outputs, batch_y, criterion)

                if not np.isnan(loss.item()):
                    train_loss.append(loss.item())
                else:
                    print(f"loss is inf at epoch {epoch + 1} and iter {i + 1}")

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                model_optim.step()

                if (i + 1) % print_batch_interval == 0:
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")

                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim,
                        scheduler,
                        epoch + 1,
                        self.args,
                        printout=True,
                    )
                    scheduler.step()

            print(f"Epoch {epoch + 1} took {time.time() - epoch_time:.4f}s")

            train_loss = np.average(train_loss)

            if len(vali_loader) == 0:
                print("No validation data")
                vali_loss = float(
                    "nan"
                )  # or you could set it to 0 or another appropriate default
            else:
                vali_loss = self.vali(vali_loader, criterion)

            if len(test_loader) == 0:
                print("No test data")
                test_loss = float(
                    "nan"
                )  # or you could set it to 0 or another appropriate default
            else:
                test_loss = self.vali(test_loader, criterion)

            print(
                f"Epoch {epoch + 1} | train loss: {train_loss:.5f} | vali loss: {vali_loss:.5f} | test loss: {test_loss:.5f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        sys.stdout.flush()
        # best_model_path = path + "/" + "checkpoint.pth"
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            self._load_model(setting)

        folder_path = "./test_results_figs/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds, trues = self._collect_predictions(test_loader)

        if preds.size == 0 or trues.size == 0:
            print(
                "Error: preds or trues arrays are empty. Check data collection logic."
            )
            return

        self._save_results(
            folder_path,
            setting,
            preds,
            trues,
            test_data.scale_y_flag,
            test_data.scaler_y,
        )

        # Mask NaN values
        nan_idx = np.isnan(trues)
        preds[nan_idx] = np.nan

        visual(
            trues,
            preds,
            os.path.join(folder_path, "before_transform"),
        )

        return

    def _prepare_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        print(
            f"debug here, _forward_model, batch_x.shape: {batch_x.shape}, batch_x_mark.shape: {batch_x_mark.shape}, , batch_y_mark.shape: {batch_y_mark.shape}"
        )
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _compute_loss(self, outputs, batch_y, criterion):
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].detach()

        # Create a mask for non-NaN values
        mask = ~torch.isnan(batch_y)

        # Apply the mask to outputs and batch_y
        outputs_masked = outputs[mask]
        batch_y_masked = batch_y[mask]

        # Calculate the loss only on the masked values
        loss = criterion(outputs_masked, batch_y_masked)
        return loss

    def _collect_predictions(self, test_loader):
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._prepare_batch(
                    batch
                )
                dec_inp = None

                outputs = self._forward_model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )

                mask = batch_y == 0

                # Apply the mask to outputs: set the corresponding positions in outputs to 0
                outputs = torch.where(mask, torch.zeros_like(outputs), outputs)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:].detach()
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].detach()
                preds.append(outputs.reshape(-1, self.args.c_out).cpu().numpy())
                trues.append(batch_y.reshape(-1, self.args.c_out).cpu().numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def _save_results(
        self, folder_path, setting, preds, trues, scale_y_flag, scaler
    ) -> None:
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"Scaled metrics - mse: {mse}, mae: {mae}")
        print(f"Scaled metrics - rmse: {rmse}, mape: {mape}, mspe: {mspe}")

        if scale_y_flag:
            (
                inversed_preds,
                inversed_trues,
                inversed_withnan_preds,
                inversed_withnan_trues,
            ) = inverse_transform(preds, trues, scaler)
            inversed_preds = np.maximum(inversed_preds, 0)
            inversed_withnan_preds = np.maximum(inversed_withnan_preds, 0)

            visual(
                inversed_withnan_trues,
                inversed_withnan_preds,
                os.path.join(folder_path, "after_transform"),
            )
            mae, mse, rmse, mape, mspe = metric(
                inversed_withnan_preds, inversed_withnan_trues
            )
            print(f"Unscaled metrics - mse: {mse}, mae: {mae}")
            print(f"Unscaled metrics - rmse: {rmse}, mape: {mape}, mspe: {mspe}")
            custom_acc = AccuracyMetricLoss(device=self.device)
            custom_acc_value = custom_acc(
                inversed_withnan_preds, inversed_withnan_trues
            )
            print(f"Unscaled custom accuracy: {custom_acc_value:.5f}")
            np.save(
                os.path.join(folder_path, "metrics.npy"),
                np.array([mae, mse, rmse, mape, mspe]),
            )
            np.save(
                os.path.join(folder_path, "inversed_pred.npy"), inversed_withnan_preds
            )
            np.save(
                os.path.join(folder_path, "inversed_true.npy"), inversed_withnan_trues
            )
        else:
            np.save(
                os.path.join(folder_path, "metrics.npy"), np.array([mae, mse, mape])
            )
            np.save(os.path.join(folder_path, "pred.npy"), preds)
            np.save(os.path.join(folder_path, "true.npy"), trues)

        with open(
            os.path.join(
                "/data/Pein/Pytorch/Wind-Solar-Prediction/output", "performance.txt"
            ),
            "a",
        ) as f:
            f.write(
                f"{setting}\nmse:{mse}, mae:{mae}, custom_acc_value:{custom_acc_value}\n\n"
            )

    def _load_model(self, setting):
        print("Loading model")
        self.model.load_state_dict(
            torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
        )

    def inference(model, x_enc, threshold=0.5):
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Forward pass through the model
            regression_outputs, classification_outputs = model(x_enc)

            # Apply threshold to classification outputs to get binary decisions
            classification_predictions = (classification_outputs > threshold).float()

            # Combine the regression and classification outputs
            final_predictions = classification_predictions * regression_outputs

        return final_predictions


class AccuracyMetricLoss(nn.Module):
    def __init__(self, device: torch.device, cap=200.0):
        super(AccuracyMetricLoss, self).__init__()
        self.cap = cap
        self.device = device

    def forward(self, pred, true):
        # Convert numpy arrays to torch tensors if necessary
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred)
        if isinstance(true, np.ndarray):
            true = torch.tensor(true)

        # Ensure both tensors are of the same shape
        assert (
            pred.shape == true.shape
        ), "Shape mismatch between prediction and ground truth arrays"

        # Move tensors to the same device
        true = true.to(self.device)
        pred = pred.to(self.device)

        # Mask out NaN values
        mask = ~torch.isnan(true)
        pred = pred[mask]
        true = true[mask]

        # If no valid values, return NaN
        if true.numel() == 0:
            print("Warning: No valid true values.")
            return torch.tensor(np.nan, device=self.device)

        # Compute S_i based on the provided conditions
        S_i = torch.where(true >= 0.2 * self.cap, true, 0.2 * self.cap)

        # Compute the custom accuracy measure
        diff = (true - pred) / S_i
        squared_diff = diff**2
        mean_squared_diff = torch.mean(squared_diff)
        accuracy_metric = (1 - torch.sqrt(mean_squared_diff)) * 100

        return accuracy_metric


def inverse_transform(preds, trues, scaler):
    # Ensure preds and trues are numpy arrays
    preds = np.array(preds)
    trues = np.array(trues)

    # Reshape for inverse transformation
    preds_reshaped = preds.reshape(-1, 1)
    trues_reshaped = trues.reshape(-1, 1)

    # Mask NaN values
    non_nan_mask = ~np.isnan(trues_reshaped)

    # Apply inverse transformation using the scaler
    preds_inverse = np.empty_like(preds_reshaped)
    trues_inverse = np.empty_like(trues_reshaped)

    # Inverse transform only non-NaN values
    preds_inverse[non_nan_mask] = scaler.inverse_transform(
        preds_reshaped[non_nan_mask].reshape(-1, 1)
    ).flatten()
    trues_inverse[non_nan_mask] = scaler.inverse_transform(
        trues_reshaped[non_nan_mask].reshape(-1, 1)
    ).flatten()

    # Restore NaN values
    preds_inverse[~non_nan_mask] = np.nan
    trues_inverse[~non_nan_mask] = np.nan

    # Flatten back the inversely transformed arrays
    inversed_preds = preds_inverse.flatten()
    inversed_trues = trues_inverse.flatten()

    # Create the inversed_withnan arrays
    inversed_withnan_preds = preds_inverse
    inversed_withnan_trues = trues_inverse

    return (
        inversed_preds[non_nan_mask.flatten()],
        inversed_trues[non_nan_mask.flatten()],
        inversed_withnan_preds,
        inversed_withnan_trues,
    )
