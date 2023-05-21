import numpy as np

from . import utils

if __name__ == '__main__':
    seed = 42
    utils.set_seed(seed)
    data = utils.jload("./data/alpaca-data-conversation.json")
    
    # random sample 20k
    N = 20000
    data_mask = np.random.choice(len(data), N, replace=False)
    data = [data[i] for i in data_mask]
    data_train = data[:10000]
    data_train = utils.jsort(data_train, key="id", integer=True)
    data_val = data[10000:]
    data_val = utils.jsort(data_val, key="id", integer=True)

    # save to json
    utils.jdump(data_train, "./data/alpaca-train-10k.json")
    utils.jdump(data_val, "./data/alpaca-val-10k.json")