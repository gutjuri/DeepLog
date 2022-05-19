import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.random import uniform
from teras.layers import LSTM
K.clear_session()


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


model = keras.models.sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())

