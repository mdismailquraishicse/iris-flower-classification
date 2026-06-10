import torch.nn as nn


class IrisFlowerClassifier(nn.Module):


    def __init__(self):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )


    def forward(self, X):

        return self.model(X)