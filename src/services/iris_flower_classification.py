import torch
import torch.nn as nn
from sklearn.datasets import load_iris


def load():

    data = load_iris()
    X, y = data.data, data.target

    print(y[:10])