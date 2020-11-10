import torch.nn as nn

class MLP(nn.Module):
    """Three-layer perceptron with ReLUs."""

    def __init__(self, n_in, n_1, n_2, n_out):
        super(MLP, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, n_1),
            nn.ReLU(),
            nn.Linear(n_1, n_2),
            nn.ReLU(),
            nn.Linear(n_2, n_out),
            nn.ReLU()
        )

    def forward(self, X):
        return self.body(X)


class RNN(nn.Module):
    """Single LSTM block with two-layer perceptron."""

    def __init__(self, n_in, num_layers, n_hidden, n_out):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hidden, num_layers, dropout=0.3)
        self.body = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(0.3),
            nn.Linear(n_hidden, n_out),
            nn.ReLU()
        )

    def forward(self, X):
        X, _ = self.lstm(X)
        return self.body(X[-1])
