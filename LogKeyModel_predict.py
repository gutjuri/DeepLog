import torch
import torch.nn as nn
import time
from LogKeyModel import Model, parseargs
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def load_labels(df, path):
    label_data = pd.read_csv(path, engine="c", na_filter=False, memory_map=True)
    label_data = label_data.set_index("id")
    label_dict = label_data["label"].to_dict()
    # print(label_dict)
    df["label"] = df["EventId"].apply(lambda x: 1 if label_dict[x] == "Anomaly" else 0)


def toList(st):
    ln = list(map(int, st.split(" ")[:-1]))
    return ln + [-1] * (window_size + 1 - len(ln))


def generate(name):
    # If you what to replicate the DeepLog paper results (Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    start_t = time.time()
    with open(name, "r") as f:
        lines = f.readlines()
        df = pd.DataFrame(
            {
                "EventId": list(map(lambda l: int(l.split("|")[0]), lines)),
                "EventSequence": list(
                    map(lambda l: toList(l.split("|", maxsplit=1)[1]), lines)
                ),
            }
        )

    end_t = time.time()
    print("Loading elapsed_time: {:.3f}s".format(end_t - start_t))
    print("Number of sessions({}): {}".format(name, len(df)))
    return df


def get_res(loader, model, device):
    res = []
    with torch.no_grad():
        for lnnr, dfl in loader.iterrows():
            sid = dfl[0]
            line = dfl[1]
            result_l = 0
            if lnnr % 1000 == 0:
                print(f"Session {lnnr}/{len(loader)} ({lnnr/len(loader)*100:.2f}%)")
            for i in range(len(line) - window_size):
                seq = line[i : i + window_size]
                label = line[i + window_size]
                seq = (
                    torch.tensor(seq, dtype=torch.float)
                    .view(-1, window_size, 1)
                    .to(device)
                )
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                # print(output)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    result_l = 1
                    break
            res.append(result_l)
    return res


if __name__ == "__main__":
    # Hyperparameters
    args = parseargs()
    num_layers = args.num_layers
    num_classes = args.num_classes
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    model_path = args.model + ".pt"

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    )

    input_size = 1  # num_classes

    model = Model(input_size, hidden_size, num_layers, num_classes, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("model_path: {}".format(model_path))
    test_normal_loader = generate(args.normal_dataset)
    # test_abnormal_loader = generate(args.abnormal_dataset)
    y_true = load_labels(test_normal_loader, args.label_path)
    # Test the model
    start_time = time.time()

    y_pred = get_res(test_normal_loader, model, device)

    # print(false_pos)

    P, R, F1, _ = precision_recall_fscore_support(
        y_pred=y_pred,
        y_true=test_normal_loader["label"].values,
        average="binary",
        pos_label=1,
    )
    elapsed_time = time.time() - start_time
    print("elapsed_time: {:.3f}s".format(elapsed_time))

    # Compute precision, recall and F1-measure

    print("Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%".format(P, R, F1))
    print("Finished Predicting")
    with open(f"results/{window_size}-{num_layers}-{hidden_size}_t.csv", "r") as f:
        t_train = float(f.readline())
    with open(f"results/{window_size}-{num_layers}-{hidden_size}-{num_candidates}.csv", "w") as f:
        f.write("Precision,Recall,F1,t_train,t_predict\n")
        f.write(f"{P:.3f},{R:.3f},{F1:.3f},{t_train:.3f},{elapsed_time:.3f}\n")
