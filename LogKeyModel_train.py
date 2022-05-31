import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os

from LogKeyModel import Model, parseargs


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    start_time = time.time()
    with open(name, "r") as f:
        for line in f.readlines():
            num_sessions += 1
            line = line.split('|')[1]
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i : i + window_size])
                outputs.append(line[i + window_size])
    end_time = time.time()
    print("Loading elapsed_time: {:.3f}s".format(end_time - start_time))
    print("Number of sessions({}): {}".format(name, num_sessions))
    print("Number of seqs({}): {}".format(name, len(inputs)))
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs)
    )
    return dataset


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2048
    model_dir = "model"

    args = parseargs()

    num_layers = args.num_layers
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    window_size = args.window_size
    name = args.training_dataset
    m_path = args.model

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )
    input_size = 1 #args.num_classes #1

    log = "Adam_batch_size={}_epoch={}".format(str(batch_size), str(num_epochs))
    model = Model(input_size, hidden_size, num_layers, num_classes, device).to(device)
    seq_dataset = generate(name)
    dataloader = DataLoader(
        seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    writer = SummaryWriter(log_dir="log/" + log, max_queue=1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        start_t_e = time.time()
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, 1).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if args.log:
                writer.add_graph(model, seq)
            writer.flush()
            # print(step)
        end_t_e = time.time()
        print(
            "Epoch [{}/{}], train_loss: {:.4f}, t_epoch: {:.3f}s, remaining: {:.3f}s".format(
                epoch + 1,
                num_epochs,
                train_loss / total_step,
                end_t_e - start_t_e,
                ((time.time() - start_time) / (epoch + 1)) * (num_epochs - epoch - 1),
            )
        )
        if args.log:
            writer.add_scalar("train_loss", train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print("elapsed_time: {:.3f}s".format(elapsed_time))
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), m_path + ".pt")
    writer.close()
    print("Finished Training")
