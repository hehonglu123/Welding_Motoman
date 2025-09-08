import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import sys, datetime, yaml, pathlib, glob, os, time, argparse
from estimate_dhdw import train_loglog
sys.path.append('../../mocap/')
from Models import *
from model_train_utils import *
from qpsolvers import solve_qp

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility
# np.random.seed(40) # for reproducibility
# torch.manual_seed(40) # for reproducibility

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print("Using device:", device)

class LayerSequenceDataset(Dataset):
    """
    Expects preprocessed per-layer dictionaries:
      {
        "scalars": FloatTensor (T, scalar_dim),
        "thermal": FloatTensor (T, thermal_dim),
        "target":  FloatTensor (T, 2)  # height, width
        # optional: "group_id" or metadata for grouped splitting
      }
    All channels should be standardized with training stats beforehand.
    """
    def __init__(self, layers_dir: List[str], sample_rate: int = 10, data_index: list = [1,2,3,6,8,9], label_index: list = [4,5]):
        super().__init__()
        self.layers_dir = layers_dir
        self.sample_rate = sample_rate
        self.data_index = data_index
        self.label_index = label_index
        self.load_layer()
    
    def load_layer(self):

        self.layers = []
        for layer_dir in self.layers_dir:
            this_layer_data = {}
            profile_welding = np.loadtxt(layer_dir+'profile_welding_'+str(self.sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
            thermal_neighborhood = np.load(layer_dir+'profile_welding_'+str(self.sample_rate)+'_thermal_neighborhood.npy')
            this_layer_data["scalars"] = torch.tensor(profile_welding[:, self.data_index], dtype=torch.float32) # x_location, cmd_v, cmd_fd, stickout, thermal_x, thermal_y
            this_layer_data["thermal"] = torch.tensor(thermal_neighborhood, dtype=torch.float32)
            this_layer_data["target"] = torch.tensor(profile_welding[:, self.label_index], dtype=torch.float32) # dh, dw
            assert this_layer_data["scalars"].shape[0] == this_layer_data["thermal"].shape[0] == this_layer_data["target"].shape[0], "Mismatch in sequence lengths"
            self.layers.append(this_layer_data)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        item = self.layers[idx]
        return {
            "scalars": item["scalars"].float(),
            "thermal": item["thermal"].float(),
            "target":  item["target"].float(),
            "meta":    item.get("meta", None)
        }

class TimeStepDataset(Dataset):
    """
    Expects preprocessed per-layer dictionaries:
      {
        "scalars": FloatTensor scalar_dim,
        "thermal": FloatTensor thermal_dim,
        "target":  FloatTensor 2  # height, width
      }
    All channels should be standardized with training stats beforehand.
    """
    def __init__(self, layers_dir: List[str], sample_rate: int = 10, data_index: list = [1,2,3,6,8,9], label_index: list = [4,5]):
        super().__init__()
        self.layers_dir = layers_dir
        self.sample_rate = sample_rate
        self.data_index = data_index
        self.label_index = label_index
        self.load_layer()

    def load_layer(self):

        self.data_samples = {}
        self.data_samples["scalars"] = []
        self.data_samples["thermal"] = []
        self.data_samples["target"] = []
        self.sample_loc = []
        thermal_lower = 7000
        thermal_upper = 28000
        feedrate_min = 50
        feedrate_max = 250
        speed_min = 0.5
        speed_max = 20
        for layer_id, layer_dir in enumerate(self.layers_dir):
            profile_welding = np.loadtxt(layer_dir+'profile_welding_'+str(self.sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
            profile_welding[:,2] = (profile_welding[:,2] - speed_min) / (speed_max - speed_min) # normalize cmd_v
            profile_welding[:,3] = (profile_welding[:,3] - feedrate_min) / (feedrate_max - feedrate_min) # normalize cmd_fd
            thermal_neighborhood = np.load(layer_dir+'profile_welding_'+str(self.sample_rate)+'_thermal_neighborhood.npy')
            # normalize thermal data
            thermal_neighborhood = (thermal_neighborhood - thermal_lower) / (thermal_upper - thermal_lower)
            self.data_samples["scalars"].append(torch.tensor(profile_welding[:, self.data_index], dtype=torch.float32)) # x_location, cmd_v, cmd_fd, stickout, thermal_x, thermal_y
            self.data_samples["thermal"].append(torch.tensor(thermal_neighborhood, dtype=torch.float32))
            self.data_samples["target"].append(torch.tensor(profile_welding[:, self.label_index], dtype=torch.float32)) # dh, dw
            self.sample_loc.extend(np.column_stack((np.ones(len(profile_welding),dtype=int)*layer_id, np.arange(len(profile_welding)))))
            assert self.data_samples["scalars"][-1].shape[0] == self.data_samples["thermal"][-1].shape[0] == self.data_samples["target"][-1].shape[0], "Mismatch in sequence lengths"
    def __len__(self):
        return len(self.data_samples["scalars"])

    def __getitem__(self, idx):
        layer_id, time_id = self.sample_loc[idx]
        return {
            "scalars": self.data_samples["scalars"][layer_id][time_id],
            "thermal": self.data_samples["thermal"][layer_id][time_id],
            "target":  self.data_samples["target"][layer_id][time_id],
        }

def collate_timesteps(batch):
    scalars = torch.stack([b["scalars"] for b in batch], dim=0).float()
    thermal = torch.stack([b["thermal"] for b in batch], dim=0).float()
    target  = torch.stack([b["target"]  for b in batch], dim=0).float()
    return scalars, thermal, target

def pad_sequences_and_make_mask(batch: List[Dict[str, Any]]
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function:
      - Pads to max T in batch
      - Returns tensors and length vector for masking
    """
    # Extract
    scalars_list = [b["scalars"] for b in batch]  # list of (T, S)
    thermal_list = [b["thermal"] for b in batch]  # list of (T, thermal_dim)
    target_list  = [b["target"]  for b in batch]  # list of (T, 2)

    lengths = torch.tensor([x.shape[0] for x in scalars_list], dtype=torch.long)

    B = len(batch)
    T_max = int(max(lengths))

    S_dim = scalars_list[0].shape[1]
    thermal_dim = thermal_list[0].shape[1]
    # Create padded tensors
    scalars_pad = torch.zeros(B, T_max, S_dim)
    thermal_pad = torch.zeros(B, T_max, thermal_dim)
    target_pad  = torch.zeros(B, T_max, 2)

    for i, (s, th, y) in enumerate(zip(scalars_list, thermal_list, target_list)):
        T = s.shape[0]
        scalars_pad[i, :T, :] = s
        thermal_pad[i, :T, :] = th
        target_pad[i,  :T, :] = y

    return scalars_pad, thermal_pad, target_pad, lengths

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, T, 2)
    lengths: (B,)
    Computes MSE only over valid timesteps.
    """
    B, T, D = pred.shape
    # build mask
    device = pred.device
    time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
    mask = (time_idx < lengths.unsqueeze(1)).float()                     # (B,T)
    mask = mask.unsqueeze(-1)                                            # (B,T,1)

    mse = (pred - target) ** 2                                           # (B,T,2)
    mse = (mse * mask).sum() / (mask.sum() * D + 1e-8)
    return mse

def masked_error(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> list:
    """
    pred, target: (B, T, 2)
    lengths: (B,)
    Computes error only over valid timesteps.
    """
    B, T, D = pred.shape
    error = []
    for b in range(B):
        L = lengths[b]
        error.extend((pred[b, :L, :].detach().cpu().numpy() - target[b, :L, :].detach().cpu().numpy()).tolist())
    
    return error

# Define hook to freeze half of W_ih
def freeze_half_weight(param: torch.Tensor, freeze_cols=[0,1]):
    mask = torch.zeros_like(param)
    mask[:, freeze_cols] = 1.0  # Freeze left half
    def hook(grad):
        return grad * (1 - mask)
    param.register_hook(hook)

def train_static(train_dataloader: DataLoader, test_dataloader: DataLoader, model: nn.Module, epochs: int, learning_rate: float, model_dir='weld_Seq_models/'):

    # loss function
    loss_fn = nn.MSELoss()
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    print_status_for_N_times = 50
    
    # Training
    training_losses = []
    testing_losses = []
    training_dh_errors_mean = []
    training_dh_errors_std = []
    training_dh_errors_95 = []
    training_dh_errors_max = []
    training_dw_errors_mean = []
    training_dw_errors_std = []
    training_dw_errors_95 = []
    training_dw_errors_max = []
    testing_dh_errors_mean = []
    testing_dh_errors_std = []
    testing_dh_errors_95 = []
    testing_dh_errors_max = []
    testing_dw_errors_mean = []
    testing_dw_errors_std = []
    testing_dw_errors_95 = []
    testing_dw_errors_max = []
    for epoch in range(epochs):
        ####### training 
        model.train()
        total_loss = 0.0
        n_batches = 0
        error_dhdw_train = []
        for step, (scalars, thermal, target) in enumerate(train_dataloader):
            scalars = scalars.to(device)
            thermal = thermal.to(device)
            target  = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(scalars, thermal)
            loss = loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            error_dhdw_train.extend((pred.detach().cpu().numpy() - target.detach().cpu().numpy()).tolist())
        this_training_loss = total_loss / max(1, n_batches)
        assert not np.isnan(this_training_loss), "Training loss is NaN!"
        training_losses.append(this_training_loss)
        ######

        ###### testing
        model.eval()
        total_loss = 0.0
        n_batches = 0
        error_dhdw_test = []
        for scalars, thermal, target in test_dataloader:
            scalars = scalars.to(device)
            thermal = thermal.to(device)
            target  = target.to(device)
            pred = model(scalars, thermal)
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            n_batches += 1
            error_dhdw_test.extend((pred.detach().cpu().numpy() - target.detach().cpu().numpy()).tolist())
        this_test_loss = total_loss / n_batches
        testing_losses.append(this_test_loss)

        # Compute error statistics
        error_dhdw_test_abs = np.abs(error_dhdw_test)
        testing_dh_errors_mean.append(np.mean(error_dhdw_test_abs[:, 0]))
        testing_dh_errors_std.append(np.std(error_dhdw_test_abs[:, 0]))
        testing_dh_errors_95.append(stats.expon(scale=np.std(error_dhdw_test_abs[:, 0])).interval(0.95)[1])
        testing_dh_errors_max.append(np.max(error_dhdw_test_abs[:, 0]))
        testing_dw_errors_mean.append(np.mean(error_dhdw_test_abs[:, 1]))
        testing_dw_errors_std.append(np.std(error_dhdw_test_abs[:, 1]))
        testing_dw_errors_95.append(stats.expon(scale=np.std(error_dhdw_test_abs[:, 1])).interval(0.95)[1])
        testing_dw_errors_max.append(np.max(error_dhdw_test_abs[:, 1]))
        error_dhdw_train_abs = np.abs(error_dhdw_train)
        training_dh_errors_mean.append(np.mean(error_dhdw_train_abs[:, 0]))
        training_dh_errors_std.append(np.std(error_dhdw_train_abs[:, 0]))
        training_dh_errors_95.append(stats.expon(scale=np.std(error_dhdw_train_abs[:, 0])).interval(0.95)[1])
        training_dh_errors_max.append(np.max(error_dhdw_train_abs[:, 0]))
        training_dw_errors_mean.append(np.mean(error_dhdw_train_abs[:, 1]))
        training_dw_errors_std.append(np.std(error_dhdw_train_abs[:, 1]))
        training_dw_errors_95.append(stats.expon(scale=np.std(error_dhdw_train_abs[:, 1])).interval(0.95)[1])
        training_dw_errors_max.append(np.max(error_dhdw_train_abs[:, 1]))

        # save the best testing model
        if epoch == 0 or this_test_loss < min(testing_losses[:-1]):
            torch.save(model.state_dict(), model_dir + 'best_model.pth')
            print(f"Epoch {epoch}: Saved new best testing model with loss {this_test_loss:.4f}")
            print("  Training dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,0]), stats.expon(scale=np.std(error_dhdw_train_abs[:,0])).interval(0.95)[1]))
            print("  Training dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,1]), stats.expon(scale=np.std(error_dhdw_train_abs[:,1])).interval(0.95)[1]))
            print("  Testing dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,0]), stats.expon(scale=np.std(error_dhdw_test_abs[:,0])).interval(0.95)[1]))
            print("  Testing dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,1]), stats.expon(scale=np.std(error_dhdw_test_abs[:,1])).interval(0.95)[1]))
            print("======")
            np.savetxt(model_dir+'training_error_dhdw_train_best_testing.csv', np.array(error_dhdw_train), delimiter=',')
            np.savetxt(model_dir+'testing_error_dhdw_test_best_testing.csv', np.array(error_dhdw_test), delimiter=',')
        # save the best training model
        if epoch == 0 or this_training_loss < min(training_losses[:-1]):
            torch.save(model.state_dict(), model_dir + 'best_training_model.pth')
            np.savetxt(model_dir+'training_error_dhdw_train_best_training.csv', np.array(error_dhdw_train), delimiter=',')
            np.savetxt(model_dir+'training_error_dhdw_test_best_training.csv', np.array(error_dhdw_test), delimiter=',')

        # print training progress
        if epoch % max(1, epochs//print_status_for_N_times) == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {this_training_loss:.4f}, Testing Loss: {this_test_loss:.4f}")
            print("  Training dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,0]), stats.expon(scale=np.std(error_dhdw_train_abs[:,0])).interval(0.95)[1]))
            print("  Training dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,1]), stats.expon(scale=np.std(error_dhdw_train_abs[:,1])).interval(0.95)[1]))
            print("  Testing dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,0]), stats.expon(scale=np.std(error_dhdw_test_abs[:,0])).interval(0.95)[1]))
            print("  Testing dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,1]), stats.expon(scale=np.std(error_dhdw_test_abs[:,1])).interval(0.95)[1]))
            print("=============================")

        # save training and testing loss every epoch
        np.savetxt(model_dir+'training_loss.csv', np.array(training_losses), delimiter=',')
        np.savetxt(model_dir+'testing_loss.csv', np.array(testing_losses), delimiter=',')
        np.savetxt(model_dir+'training_dh_errors.csv', np.column_stack((training_dh_errors_mean, training_dh_errors_std, training_dh_errors_95, training_dh_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'training_dw_errors.csv', np.column_stack((training_dw_errors_mean, training_dw_errors_std, training_dw_errors_95, training_dw_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'testing_dh_errors.csv', np.column_stack((testing_dh_errors_mean, testing_dh_errors_std, testing_dh_errors_95, testing_dh_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'testing_dw_errors.csv', np.column_stack((testing_dw_errors_mean, testing_dw_errors_std, testing_dw_errors_95, testing_dw_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
    return model, training_losses, testing_losses

def train_RNN(train_dataloader: DataLoader, test_dataloader: DataLoader, model: nn.Module, epochs: int, learning_rate: float, model_dir='weld_Seq_models/'):

    # loss function
    loss_fn = masked_mse_loss
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    print_status_for_N_times = 50
    
    scaler = GradScaler(device=device_name)
    acc_steps = 32          # tune this
    max_norm = 1.0         # grad clip

    # Training
    training_losses = []
    testing_losses = []
    training_dh_errors_mean = []
    training_dh_errors_std = []
    training_dh_errors_95 = []
    training_dh_errors_max = []
    training_dw_errors_mean = []
    training_dw_errors_std = []
    training_dw_errors_95 = []
    training_dw_errors_max = []
    testing_dh_errors_mean = []
    testing_dh_errors_std = []
    testing_dh_errors_95 = []
    testing_dh_errors_max = []
    testing_dw_errors_mean = []
    testing_dw_errors_std = []
    testing_dw_errors_95 = []
    testing_dw_errors_max = []

    for epoch in range(epochs):
        ####### training 
        model.train()
        total_loss = 0.0
        n_steps = 0
        error_dhdw_train = []
        optimizer.zero_grad(set_to_none=True)
        for step, (scalars, thermal, target, lengths) in enumerate(train_dataloader):
            scalars = scalars.to(device)
            thermal = thermal.to(device)
            target  = target.to(device)
            lengths = lengths.to(device)

            with autocast(device_type=device_name, dtype=torch.float16):
                pred = model(scalars, thermal, lengths)
                loss = masked_mse_loss(pred, target, lengths) / acc_steps
                error_dhdw_train.extend(masked_error(pred, target, lengths))

            scaler.scale(loss).backward()

            if (step + 1) % acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * acc_steps
            n_steps += 1
        # flush leftover grads
        rem = n_steps % acc_steps
        if rem != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        this_training_loss = total_loss / max(1, n_steps)
        assert not np.isnan(this_training_loss), "Training loss is NaN!"
        training_losses.append(this_training_loss)
        ######

        ###### testing
        model.eval()
        total_loss = 0.0
        n_batches = 0
        error_dhdw_test = []
        for scalars, thermal, target, lengths in test_dataloader:
            scalars = scalars.to(device)
            thermal = thermal.to(device)
            target  = target.to(device)
            lengths = lengths.to(device)
            pred = model(scalars, thermal, lengths)
            loss = loss_fn(pred, target, lengths)
            total_loss += loss.item()
            n_batches += 1
            error_dhdw_test.extend(masked_error(pred, target, lengths))
        this_test_loss = total_loss / n_batches
        testing_losses.append(this_test_loss)

        # Compute error statistics
        error_dhdw_test_abs = np.abs(error_dhdw_test)
        testing_dh_errors_mean.append(np.mean(error_dhdw_test_abs[:,0]))
        testing_dh_errors_std.append(np.std(error_dhdw_test_abs[:,0]))
        testing_dh_errors_95.append(stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,0]))).interval(0.95)[1])
        testing_dh_errors_max.append(np.max(error_dhdw_test_abs[:,0]))
        testing_dw_errors_mean.append(np.mean(error_dhdw_test_abs[:,1]))
        testing_dw_errors_std.append(np.std(error_dhdw_test_abs[:,1]))
        testing_dw_errors_95.append(stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,1]))).interval(0.95)[1])
        testing_dw_errors_max.append(np.max(error_dhdw_test_abs[:,1]))
        error_dhdw_train_abs = np.abs(error_dhdw_train)
        training_dh_errors_mean.append(np.mean(error_dhdw_train_abs[:,0]))
        training_dh_errors_std.append(np.std(error_dhdw_train_abs[:,0]))
        training_dh_errors_95.append(stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,0]))).interval(0.95)[1])
        training_dh_errors_max.append(np.max(error_dhdw_train_abs[:,0]))
        training_dw_errors_mean.append(np.mean(error_dhdw_train_abs[:,1]))
        training_dw_errors_std.append(np.std(error_dhdw_train_abs[:,1]))
        training_dw_errors_95.append(stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,1]))).interval(0.95)[1])
        training_dw_errors_max.append(np.max(error_dhdw_train_abs[:,1]))

        # save the best testing model
        if epoch == 0 or this_test_loss < min(testing_losses[:-1]):
            torch.save(model.state_dict(), model_dir + 'best_model.pth')
            print(f"Epoch {epoch}: Saved new best testing model with loss {this_test_loss:.4f}")
            print("  Training dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,0]), stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,0]))).interval(0.95)[1]))
            print("  Training dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,1]), stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,1]))).interval(0.95)[1]))
            print("  Testing dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,0]), stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,0]))).interval(0.95)[1]))
            print("  Testing dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,1]), stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,1]))).interval(0.95)[1]))
            print("======")
            np.savetxt(model_dir+'training_error_dhdw_train_best_testing.csv', np.array(error_dhdw_train), delimiter=',')
            np.savetxt(model_dir+'testing_error_dhdw_test_best_testing.csv', np.array(error_dhdw_test), delimiter=',')

        # save the best training model
        if epoch == 0 or this_training_loss < min(training_losses[:-1]):
            torch.save(model.state_dict(), model_dir + 'best_training_model.pth')
            np.savetxt(model_dir+'training_error_dhdw_train_best_training.csv', np.array(error_dhdw_train), delimiter=',')
            np.savetxt(model_dir+'training_error_dhdw_test_best_training.csv', np.array(error_dhdw_test), delimiter=',')

        # print training progress
        if epoch % max(1, epochs//print_status_for_N_times) == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {this_training_loss:.4f}, Testing Loss: {this_test_loss:.4f}")
            print("  Training dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,0]), stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,0]))).interval(0.95)[1]))
            print("  Training dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_train_abs[:,1]), stats.expon(scale=np.std(np.abs(error_dhdw_train_abs[:,1]))).interval(0.95)[1]))
            print("  Testing dh error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,0]), stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,0]))).interval(0.95)[1]))
            print("  Testing dw error: mean {:.4f}, 95% {:.4f}".format(np.mean(error_dhdw_test_abs[:,1]), stats.expon(scale=np.std(np.abs(error_dhdw_test_abs[:,1]))).interval(0.95)[1]))
            print("=============================")

        # save training and testing loss every epoch
        np.savetxt(model_dir+'training_loss.csv', np.array(training_losses), delimiter=',')
        np.savetxt(model_dir+'testing_loss.csv', np.array(testing_losses), delimiter=',')
        np.savetxt(model_dir+'training_dh_errors.csv', np.column_stack((training_dh_errors_mean, training_dh_errors_std, training_dh_errors_95, training_dh_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'training_dw_errors.csv', np.column_stack((training_dw_errors_mean, training_dw_errors_std, training_dw_errors_95, training_dw_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'testing_dh_errors.csv', np.column_stack((testing_dh_errors_mean, testing_dh_errors_std, testing_dh_errors_95, testing_dh_errors_max)), delimiter=',', header='mean,std,95,max', comments='')
        np.savetxt(model_dir+'testing_dw_errors.csv', np.column_stack((testing_dw_errors_mean, testing_dw_errors_std, testing_dw_errors_95, testing_dw_errors_max)), delimiter=',', header='mean,std,95,max', comments='')

    return model, training_losses, testing_losses

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compare RNN weights from PyTorch models")
    parser.add_argument("--train", action='store_true', help="Train the model or not, default is False")
    parser.add_argument("--load_pretrained", action='store_true', help="Load pre-trained model or not, default is False")
    parser.add_argument("--load_model_dir", type=str, default='', help="Directory to load the model from")
    parser.add_argument("--all_data_testing", action='store_true', help="Use all data for testing or not, default is False")
    parser.add_argument("--model_type", type=str, default='WAAM_GRU', help="Model type: WAAM_GRU, WAAM_NN")
    parser.add_argument("--thermal_emb", type=int, default=64, help="Embedding size for thermal data, default is 64")
    parser.add_argument("--scalar_emb", type=int, default=32, help="Embedding size for scalar data, default is 32")
    parser.add_argument("--nn_layers", type=int, default=1, help="Number of RNN/NN layers, default is 1")
    parser.add_argument("--nn_hidden_size", type=int, default=128, help="Model hidden size, default is 128")
    parser.add_argument("--innovation", action='store_true', help="Innovation model or not, default is False")
    parser.add_argument("--sample_rate", type=int, default=10, help="Sample rate for the data, default is 10")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs for training, default is 5000")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer, default is 0.001")
    parser.add_argument("--no_feat_x_location", action='store_true', help="Not use x location as input feature, default is False")
    parser.add_argument("--no_feat_stickout", action='store_true', help="Not use stickout as input feature, default is False")
    parser.add_argument("--no_feat_thermal_x", action='store_true', help="Not use thermal x as input feature, default is False")
    parser.add_argument("--no_feat_thermal_y", action='store_true', help="Not use thermal y as input feature, default is False")
    parser.add_argument("--no_feat_neighbor_thermal", action='store_true', help="Not use neighbor thermal as input feature, default is False")
    parse_arg = parser.parse_args()

    # load data
    geo_data_dir = '../../data/wall_weld_test/'
    logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']
    # logdata_dir_all = ['weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']

    train_flag = parse_arg.train # set to False to use the pre-trained model
    load_pretrained = parse_arg.load_pretrained
    use_all_data_for_testing = parse_arg.all_data_testing # set to True to use all data for testing, otherwise use the last tote for testing

    model_dir = 'weld_Seq_models/' # directory to save the model
    # model directory
    if train_flag and not load_pretrained:

        #### parameters
        # model type
        model_type = parse_arg.model_type # 'WAAM_GRU', 'WAAM_NN'
        if model_type not in ['WAAM_GRU', 'WAAM_NN']:
            print("Invalid model type:", model_type, ". Please choose from 'WAAM_GRU', 'WAAM_NN'.")
            sys.exit(1)
        # input features
        feat_x_location = not parse_arg.no_feat_x_location # use x location as input feature
        feat_cmd_v = True # use command velocity as input feature
        feat_cmd_fd = True # use command feed as input feature
        feat_stickout = not parse_arg.no_feat_stickout # use stickout length as input feature
        feat_thermal_x = not parse_arg.no_feat_thermal_x # use thermal data as input feature
        feat_thermal_y = not parse_arg.no_feat_thermal_y # use thermal data as input feature
        feat_neighbor_thermal = not parse_arg.no_feat_neighbor_thermal # use neighborhood thermal data as input feature
        # model structure
        innovation = parse_arg.innovation # default is True, set to False for closed loop model
        if 'GRU' in model_type:
            rnn_hidden_size = parse_arg.nn_hidden_size # default is 128
            if rnn_hidden_size < 1:
                print("Invalid RNN hidden size. Please provide a value greater than or equal to 1.")
                sys.exit(1)
            rnn_layers = parse_arg.nn_layers # number of RNN layers
        else:
            nn_hidden_size = parse_arg.nn_hidden_size # default is 128
            if nn_hidden_size < 1:
                print("Invalid NN hidden size. Please provide a value greater than or equal to 1.")
                sys.exit(1)
            nn_layers = parse_arg.nn_layers # number of NN layers
        thermal_emb = parse_arg.thermal_emb # embedding size for thermal data
        scalar_emb = parse_arg.scalar_emb # embedding size for scalar data
        model_output_size = 2 # dh, dw
        # data processing parameters
        sample_rate = parse_arg.sample_rate # Hz, using the rate of fronious control
        train_test_split = 0.8 # 80% for training, 20% for testing
        # learning parameters
        epochs = parse_arg.epochs # number of epochs for training
        learning_rate = parse_arg.learning_rate # learning rate for training

        training_params = {
            'geo_data_dir': geo_data_dir, 'logdata_dir_all': logdata_dir_all,
            'model_type': model_type,
            'feat_x_location': feat_x_location, 'feat_cmd_v': feat_cmd_v, 'feat_cmd_fd': feat_cmd_fd,
            'feat_stickout': feat_stickout, 'feat_thermal_x': feat_thermal_x, 'feat_thermal_y': feat_thermal_y,
            'feat_neighbor_thermal': feat_neighbor_thermal,
            'innovation': innovation, 'thermal_emb': thermal_emb, 'scalar_emb': scalar_emb,
            'model_output_size': model_output_size,
            'sample_rate': sample_rate, 'train_test_split': train_test_split, 
            'epochs': epochs,'learning_rate': learning_rate
        }
        if 'GRU' in model_type:
            training_params['rnn_hidden_size'] = rnn_hidden_size
            training_params['rnn_layers'] = rnn_layers
        else:
            training_params['nn_hidden_size'] = nn_hidden_size
            training_params['nn_layers'] = nn_layers
        # save the training parameters
        # add timestamp to the model_dir
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_dir = model_dir + "model_"+ timestamp + '/'
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        with open(model_dir+'training_params.yaml', 'w') as f:
            yaml.dump(training_params, f, default_flow_style=False)
    else:
        pre_trained_model_dir = model_dir+'model_20250715_151650/' if parse_arg.load_model_dir == '' else model_dir+parse_arg.load_model_dir+'/'
        model_dir = deepcopy(pre_trained_model_dir) # use the pre-trained model directory

        # load the training parameters
        with open(pre_trained_model_dir+'training_params.yaml', 'r') as f:
            training_params = yaml.safe_load(f)
        geo_data_dir = training_params['geo_data_dir']
        logdata_dir_all = training_params['logdata_dir_all']
        model_type = training_params['model_type']
        feat_x_location = training_params['feat_x_location']
        feat_cmd_v = training_params['feat_cmd_v']
        feat_cmd_fd = training_params['feat_cmd_fd']
        feat_stickout = training_params['feat_stickout']
        feat_thermal_x = training_params['feat_thermal_x']
        feat_thermal_y = training_params['feat_thermal_y']
        feat_neighbor_thermal = training_params['feat_neighbor_thermal']
        innovation = training_params['innovation']
        rnn_hidden_size = training_params['rnn_hidden_size']
        rnn_layers = training_params['rnn_layers']
        thermal_emb = training_params['thermal_emb']
        scalar_emb = training_params['scalar_emb']
        model_output_size = training_params['model_output_size']
        sample_rate = training_params['sample_rate']
        train_test_split = training_params['train_test_split']
        epochs = training_params['epochs']
        learning_rate = training_params['learning_rate']

        if train_flag:
            # save the training parameters
            # add timestamp to the model_dir
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            model_dir = "weld_Seq_models/model_"+ timestamp + '/'
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
            with open(model_dir+'training_params.yaml', 'w') as f:
                yaml.dump(training_params, f, default_flow_style=False)

    save_loglog = False # set to True to save the log-log model parameters

    print("Training parameters:")
    print("Train flag:", train_flag)
    print("Model directory:", model_dir)
    if (not train_flag) or load_pretrained:
        print("Using pre-trained model:", pre_trained_model_dir)
    print("Model type:", model_type)
    print("Features")
    print("  x_location:", feat_x_location)
    print("  cmd_v:", feat_cmd_v)
    print("  cmd_fd:", feat_cmd_fd)
    print("  stickout:", feat_stickout)
    print("  thermal_x:", feat_thermal_x)
    print("  thermal_y:", feat_thermal_y)
    print("  neighbor_thermal:", feat_neighbor_thermal)
    print("Model structure:")
    if 'GRU' in model_type:
        print("  RNN hidden size:", rnn_hidden_size)
        print("  RNN layers:", rnn_layers)
    else:
        print("  NN hidden size:", nn_hidden_size)
        print("  NN layers:", nn_layers)
    print("  Thermal embedding:", thermal_emb)
    print("  Scalar embedding:", scalar_emb)
    print("  Model output size:", model_output_size)
    print("Training parameters:")
    print("  Epochs:", epochs)
    print("  Learning rate:", learning_rate)

    ### data processing
    data_dirs = []
    train_data_batch_len = []
    for i, logdata_dir in enumerate(logdata_dir_all):
        total_layers_name = glob.glob(geo_data_dir+logdata_dir+'layer*')
        # get printed layer number
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
        for layer_n_id, layer_n in enumerate(layer_nums):
            layer_name = 'layer'+str(layer_n)
            this_layer_dir = geo_data_dir + logdata_dir + layer_name + '/'
            data_dirs.append(this_layer_dir)
            this_layer = np.loadtxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
            train_data_batch_len.append(len(this_layer))
    
    print(f'Min length of the data: {np.min(train_data_batch_len)}, max length of the data: {np.max(train_data_batch_len)}')
    # total amount of data
    print("Total amount of data: ", len(data_dirs))
    print("Total amount of data batch: ", np.sum(train_data_batch_len))
    # spread the data randomly into 5 totes but with almost equal amount of data
    # spread_epsilon = 0.8
    train_data_split_dir = [[] for _ in range(5)]
    train_data_split_len = np.array([0 for _ in range(5)])
    for dir_i, data_dir in enumerate(data_dirs):
        if np.any(train_data_split_len==0):
            # equally distribute the data to the totes with zero length
            zero_totes = np.where(train_data_split_len==0)[0]
            chosen_tote = np.random.choice(zero_totes)
        else:
            # randomly choose a tote based on the inverse of the current length of the tote
            choice_p = 1/train_data_split_len
            choice_p /= np.sum(choice_p)  # normalize to sum to 1
            chosen_tote = np.random.choice(np.arange(5), p=choice_p)
        train_data_split_dir[chosen_tote].append(data_dir)
        train_data_split_len[chosen_tote] += train_data_batch_len[dir_i]
    print("Total amount of data in each tote: ", train_data_split_len)
    print("min ratio:", np.min(train_data_split_len)/np.sum(train_data_split_len))
    print("max ratio:", np.max(train_data_split_len)/np.sum(train_data_split_len))

    ###### load data ######
    train_data_dir_tote = train_data_split_dir[:-1]
    if not use_all_data_for_testing:
        test_data_dir_tote = train_data_split_dir[-1:]
    else:
        test_data_dir_tote = deepcopy(train_data_split_dir)
    train_data_dir_tote = [item for sublist in train_data_dir_tote for item in sublist]
    test_data_dir_tote = [item for sublist in test_data_dir_tote for item in sublist]

    print("Load to datasets...")
    data_index = []
    data_index.append(1) if feat_x_location else None
    data_index.append(2) if feat_cmd_v else None
    data_index.append(3) if feat_cmd_fd else None
    data_index.append(6) if feat_stickout else None
    data_index.append(8) if feat_thermal_x else None
    data_index.append(9) if feat_thermal_y else None

    if 'GRU' in model_type:
        train_ds = LayerSequenceDataset(train_data_dir_tote, sample_rate=sample_rate, data_index=data_index, label_index=[4,5])
        test_ds = LayerSequenceDataset(test_data_dir_tote, sample_rate=sample_rate, data_index=data_index, label_index=[4,5])
        train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=pad_sequences_and_make_mask)
        test_dataloader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=pad_sequences_and_make_mask)
    else:
        train_ds = TimeStepDataset(train_data_dir_tote, sample_rate=sample_rate, data_index=data_index, label_index=[4,5])
        test_ds = TimeStepDataset(test_data_dir_tote, sample_rate=sample_rate, data_index=data_index, label_index=[4,5])
        train_dataloader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True, collate_fn=collate_timesteps)
        test_dataloader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=collate_timesteps)

    if 'GRU' in model_type:
        model = WAAMGRUModel(scalar_dim=len(data_index), use_thermal=feat_neighbor_thermal, thermal_emb=thermal_emb, scalar_emb=scalar_emb, rnn_hidden=rnn_hidden_size, rnn_layers=rnn_layers).to(device)
    else:
        model = WAAMNNModel(scalar_dim=len(data_index), use_thermal=feat_neighbor_thermal, thermal_emb=thermal_emb, scalar_emb=scalar_emb, nn_hidden=nn_hidden_size, nn_layers=nn_layers).to(device)
    print("Model trainable parameters:",count_parameters(model))

    if train_flag:
        if load_pretrained:
            pass
        # training loop
        start_time = time.time()
        if 'GRU' in model_type:
            _, training_loss, testing_loss = train_RNN(train_dataloader, test_dataloader, model, epochs, learning_rate, model_dir=model_dir)
        else:
            _, training_loss, testing_loss = train_static(train_dataloader, test_dataloader, model, epochs, learning_rate, model_dir=model_dir)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        model.load_state_dict(torch.load(model_dir+'best_model.pth',weights_only=True)) # load the best model for evaluation
        # save loss
        np.savetxt(model_dir+'training_loss.csv', training_loss, delimiter=',')
        np.savetxt(model_dir+'testing_loss.csv', testing_loss, delimiter=',')
        np.savetxt(model_dir+'training_time.csv', np.array([end_time - start_time]), delimiter=',')
    else:
        # load the pre-trained model
        model.load_state_dict(torch.load(model_dir+'best_model.pth',weights_only=True))

        # test_u = np.zeros((1,1000,2))
        # test_u = torch.tensor(test_u, dtype=torch.float32).to(device)
        # test_x = np.ones((1,1000,2))*2 # 1 sample, 1000 time steps, model_input_size features
        # test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        # model.eval()
        # with torch.no_grad():
        #     test_y = model(test_x, test_u)
        # test_y = test_y.cpu().numpy().astype(np.float64)
        # plt.plot(test_y[0,:,0], label='dh prediction')
        # plt.plot(test_y[0,:,1], label='dw prediction')
        # plt.xlabel('Time Step', fontsize=xy_label_size)
        # plt.ylabel('Prediction', fontsize=xy_label_size)
        # plt.title('Model Prediction', fontsize=title_size)
        # plt.legend(fontsize=legend_size)
        # plt.show()

        print("Loaded pre-trained model from: ", model_dir+'best_model.pth')
        training_loss = np.loadtxt(model_dir+'training_loss.csv', delimiter=',')
        testing_loss = np.loadtxt(model_dir+'testing_loss.csv', delimiter=',')

        # visualize the parameters
        if viz_weightings and model_type == 'RNN':
            open_loop_string = 'Open Loop' if open_loop else 'Closed Loop'
            if open_loop:
                Whh = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
                Wih = model.rnn.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
                bh = model.rnn.bias_hh_l0.detach().cpu().numpy().astype(np.float64)
                bi = model.rnn.bias_ih_l0.detach().cpu().numpy().astype(np.float64)
            else:
                Whh = model.rnn_cell.weight_hh.detach().cpu().numpy().astype(np.float64)
                Wih = model.rnn_cell.weight_ih.detach().cpu().numpy().astype(np.float64)
                bh = model.rnn_cell.bias_hh.detach().cpu().numpy().astype(np.float64)
                bi = model.rnn_cell.bias_ih.detach().cpu().numpy().astype(np.float64)
            Woh = model.fc.weight.detach().cpu().numpy().astype(np.float64)
            bo = model.fc.bias.detach().cpu().numpy().astype(np.float64)

            print("Whh shape:", Whh.shape, "Wih shape:", Wih.shape, "bh shape:", bh.shape, "bi shape:", bi.shape)
            print("Woh shape:", Woh.shape, "bo shape:", bo.shape)

            # find the equilibrium point of the RNN

            # find the matrix elimited the parameter redundancy
            W_hh_min = Whh + Wih[:,-2:]@Woh

            if not open_loop:
                prediction,zt = model.forward_linear_mat(test_data_labels, test_data_input)
                prediction_true = model(test_data_labels, test_data_input)

                # make sure prediction and prediction_true are the same
                assert np.allclose(prediction.detach().cpu().numpy(), prediction_true.detach().cpu().numpy()), "Prediction and prediction_true are not the same!"

                zt_flatten = zt.view(-1).detach().cpu().numpy()
                zt_tanh_derivative = 1 - np.tanh(zt_flatten)**2
                # plot zt_tanh_derivative distribution
                plt.figure(figsize=(10, 5))
                plt.hist(zt_tanh_derivative, bins=100, density=True, alpha=0.7, color='blue')
                plt.title('Distribution of $z_t$ Tanh Derivative', fontsize=title_size)
                plt.xlabel('$z_t$ Tanh Derivative', fontsize=xy_label_size)
                plt.ylabel('Density', fontsize=xy_label_size)
                plt.xticks(fontsize=xy_tick_size)
                plt.yticks(fontsize=xy_tick_size)
                plt.grid()
                plt.tight_layout()
                plt.show()

            # plot Whh and Wih
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 4, 1)
            plt.imshow(Whh, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{hh}}$ Matrix', fontsize=title_size)
            plt.xlabel('Hidden Units', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.subplot(1, 4, 2)
            plt.imshow(Wih, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{ih}}$ Matrix', fontsize=title_size)
            plt.xlabel('Input Features', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.suptitle(f'RNN Weight Matrices {open_loop_string}', fontsize=sup_title_size)
            plt.subplot(1, 4, 3)
            plt.imshow(W_hh_min, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{hh,min}}$ Matrix', fontsize=title_size)
            plt.xlabel('Hidden Units', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            # plt.subplot(1, 4, 3)
            # plt.imshow(bh.reshape(-1, 1), cmap='viridis', aspect='equal')
            # plt.colorbar()
            # plt.title(f'RNN $b_{{h}}$ Vector', fontsize=title_size)
            # plt.xlabel('Hidden Units', fontsize=xy_label_size)
            # plt.ylabel('Bias', fontsize=xy_label_size)
            # plt.xticks(fontsize=xy_tick_size)
            # plt.yticks(fontsize=xy_tick_size)
            # plt.subplot(1, 4, 4)
            # plt.imshow(bi.reshape(-1, 1), cmap='viridis', aspect='equal')
            # plt.colorbar()
            # plt.title(f'RNN $b_{{i}}$ Vector', fontsize=title_size)
            # plt.xlabel('Input Features', fontsize=xy_label_size)
            # plt.ylabel('Bias', fontsize=xy_label_size)
            # plt.xticks(fontsize=xy_tick_size)
            # plt.yticks(fontsize=xy_tick_size)
            plt.tight_layout()
            plt.show()

            # eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(Whh)
            plt.plot(np.sort(np.abs(eigenvalues))[::-1], 'o')
            plt.title(f'Eigenvalues of RNN $W_{{hh}}$ Matrix ({open_loop_string})', fontsize=title_size)
            plt.xlabel('Index', fontsize=xy_label_size)
            plt.ylabel('Eigenvalue Magnitude', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.grid()
            plt.tight_layout()
            plt.show()

            eigenvalues, eigenvectors = np.linalg.eig(W_hh_min)
            plt.plot(np.sort(np.abs(eigenvalues))[::-1], 'o')
            plt.title(f'Eigenvalues of RNN $W_{{hh}}$ Matrix ({open_loop_string})', fontsize=title_size)
            plt.xlabel('Index', fontsize=xy_label_size)
            plt.ylabel('Eigenvalue Magnitude', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.grid()
            plt.tight_layout()
            plt.show()

    # plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(testing_loss, label='Testing Loss')
    plt.xlabel('Epochs', fontsize=xy_label_size)
    plt.ylabel('Loss', fontsize=xy_label_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.grid()
    plt.legend(fontsize=legend_size)
    plt.title('Training and Testing Loss', fontsize=title_size)
    plt.tight_layout()
    plt.savefig(model_dir+'training_testing_loss.png')
    # plt.show()