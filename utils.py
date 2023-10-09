import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
import torch.nn as nn
from scipy import signal

# https://johannfaouzi.github.io/pyts/auto_examples/plot_rp.html?highlight=recurrent
# https://pyts.readthedocs.io/en/stable/auto_examples/multivariate/plot_joint_rp.html#sphx-glr-auto-examples-multivariate-plot-joint-rp-py
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from sklearn.metrics import roc_curve, roc_auc_score
from PIL import Image

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def evaluation(performance, verbose=True):

    average_p = np.mean([val[1] for val in performance])
    average_r = np.mean([val[2] for val in performance])
    average_f1 = (average_p * average_r) * 2 / (average_p + average_r)

    if verbose:
        print("average precision: ", average_p)
        print("average recall: ", average_r)
        print("best f1: ", average_f1)
    return average_p, average_r, average_f1
		
class output:
    def __init__(
        self,
        entity_name, 
        train_loader,
        #val_loader,
        test_loader,
        windows_label_compressed,
        normal_shape_1,
    ):
        self.entity_name = entity_name
        self.train_loader = train_loader
        #self.val_loader = val_loader
        self.test_loader = test_loader
        self.labels = windows_label_compressed
        self.data_dim = normal_shape_1
		
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="./model/checkpoint.pt"):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        #torch.save(model.state_dict(), self.path)
        torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
        }, self.path)
        self.val_loss_min = val_loss
