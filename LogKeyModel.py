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
        #x = F.one_hot(x.to(torch.int64), self.num_keys).to(torch.float)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, X, y=None, k=1, variable=False, verbose=True):
        """Predict the k most likely output values
            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)
            y : Ignored
                Ignored
            k : int, default=1
                Number of output items to generate
            variable : boolean, default=False
                If True, predict inputs of different sequence lengths
            verbose : boolean, default=True
                If True, print output
            Returns
            -------
            result : torch.Tensor of shape=(n_samples, k)
                k most likely outputs
            confidence : torch.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        # Get the predictions
        result = super().predict(X, variable=variable, verbose=verbose)
        # Get the probabilities from the log probabilities
        result = result.exp()
        # Compute k most likely outputs
        confidence, result = result.topk(k)
        # Return result
        return result, confidence

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
    parser.add_argument('-label_path', type=str)

    parser.add_argument('-cuda', default=True, type=bool)
    parser.add_argument('-log', default=True, type=bool)
    parser.add_argument('-model', default='model/Adam_batch_size=2048_epoch=300.pt', type=str)

    args = parser.parse_args()
    return args