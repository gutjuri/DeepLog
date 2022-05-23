import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from TCN import TCN


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys, device):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #TCN(input_size, hidden_size, [10], 2, 0.2)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.device = device

    def forward(self, x):
        x = F.one_hot(x.to(torch.int64), self.num_keys).to(torch.float)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-num_classes', default=30, type=int)
    parser.add_argument('-num_epochs', default=300, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-training_dataset', default="data/hdfs_train", type=str)

    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-normal_dataset', default='data/hdfs_test_normal', type=str)
    parser.add_argument('-abnormal_dataset', default='data/hdfs_test_abnormal', type=str)

    parser.add_argument('-cuda', default=True, type=bool)
    parser.add_argument('-log', default=True, type=bool)
    parser.add_argument('-model', default='model/Adam_batch_size=2048_epoch=300.pt', type=str)

    args = parser.parse_args()
    return args