import torch
import torch.nn as nn
import time
import argparse

from LogKeyModel import Model, parseargs

def generate(name):
    # If you what to replicate the DeepLog paper results (Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = set()
    # hdfs = []
    with open(name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

def count_positives(loader, model, device):
    positives = 0
    with torch.no_grad():
        for line in loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    positives += 1
    return positives

if __name__ == '__main__':
    # Hyperparameters
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'
    
    args = parseargs()
    num_layers = args.num_layers
    num_classes = args.num_classes
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    model = Model(input_size, hidden_size, num_layers, num_classes, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate(args.normal_dataset)
    test_abnormal_loader = generate(args.abnormal_dataset)

    # Test the model
    start_time = time.time()
    FP = count_positives(test_normal_loader, model, device)
    TP = 1# count_positives(test_abnormal_loader, model, device)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
