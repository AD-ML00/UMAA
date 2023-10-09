import numpy as np
import pandas as pd
import torch.utils.data as data_utils
import torch
from utils import *
import csv
import os
import ast
from sklearn import preprocessing

def SMAP_MSL_processor(dataset, dataset_folder):
    with open(os.path.join(dataset_folder, "labeled_anomalies.csv"), "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
    return data_info
	
def get_create_window(args, normal, attack, labels):

    windows_normal = normal.values[np.arange(args.window)[None, :] + np.arange(normal.shape[0] - args.window)[:, None]]
    windows_attack = attack.values[np.arange(args.window)[None, :] + np.arange(attack.shape[0] - args.window)[:, None]]
    windows_label = labels.values[np.arange(args.window)[None, :] + np.arange(labels.shape[0] - args.window)[:, None]]
		
    windows_label_compressed = np.array([1.0 if (np.sum(window) > 0) else 0 for window in windows_label])

    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal).float()
    ), batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float()
    ), batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, windows_label_compressed


def preprocess(df):
    """returns normalized and standardized data."""
    min_max_scaler = preprocessing.MinMaxScaler()
	
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be a 2-D array")

    if(np.isnan(df).any()):
        print("X The Array contain NaN values")
        df = np.nan_to_num(df)
		
    # normalize data
    df = min_max_scaler.fit_transform(df)

    df = pd.DataFrame(df)

    return df
	
	
def get_data_loader(args):

    min_max_scaler = preprocessing.MinMaxScaler()
	
    if args.dname in ['SWAT', 'WADI']:
        normal = pd.read_pickle(f'./dataset/{args.dname}/train_m_index.p')
        attack = pd.read_pickle(f'./dataset/{args.dname}/test_m_index.p') 
        labels = pd.read_pickle(f'./dataset/{args.dname}/labels_m_index.p') 	

        if args.dname in ["WADI"]:
            normal = preprocess(normal)
            attack = preprocess(attack)
            labels = preprocess(labels)

        train_loader, test_loader, windows_label_compressed = get_create_window(args, normal, attack, labels)
        out = output(args.dname, train_loader, test_loader, windows_label_compressed, normal.shape[1])
        yield out

    elif args.dname in ['MSL', 'SMAP']:
        filepath = os.getcwd() + "/data/SMAP_MSL/"
        data_info = SMAP_MSL_processor(args.dname, filepath)

        for row in data_info:
			
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.float32)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True

            x_train_name = filepath + "train/" + row[0] + ".npy"
            normal_temp = np.load(x_train_name)
            x_test_name = filepath + "test/" + row[0] + ".npy"
            attack_temp = np.load(x_test_name)
            label_temp = label

            normal_temp = np.asarray(normal_temp, dtype=np.float32)
            attack_temp = np.asarray(attack_temp, dtype=np.float32)
            label_temp = np.asarray(label_temp, dtype=np.float32)
			
            normal_scaled = min_max_scaler.fit_transform(normal_temp)
            attack_scaled = min_max_scaler.transform(attack_temp)
			
            normal = pd.DataFrame(normal_scaled)
            attack = pd.DataFrame(attack_scaled)
            labels = pd.DataFrame(label_temp)

            train_loader, test_loader, windows_label_compressed = get_create_window(args, normal, attack, labels)
            out = output(row[0], train_loader, test_loader, windows_label_compressed, normal.shape[1])
            yield out


