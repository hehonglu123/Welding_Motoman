import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import sys, datetime, yaml, pathlib, glob

def train():

    pass

if __name__ == "__main__":

    # load data
    data_dir = '../../data/wall_weld_test/'
    logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']

    # parameters
    sample_rate = 30 # Hz, using the rate of ir camera
    train_test_split = 0.8 # 80% for training, 20% for testing
    epochs = 10000 # number of epochs for training

    ignore_start_end = 5
    start_x = -55 + ignore_start_end
    end_x = 55 - ignore_start_end

    # prepare data
    train_data_dirs = []
    test_data_dirs = []
    