"""Research model definitions extracted without notebook execution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DFFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model1(nn.Module):
    def __init__(self, n_features, dropout):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))  # Only pass x through fc1 once here
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.sigmoid(self.fc5(x))

        return x



class Model4(nn.Module):
    def __init__(self, n_features):
        super(Model4, self).__init__()
        self.lstm = nn.LSTM(n_features, 65, num_layers=2, dropout=0.78, batch_first=True)
        self.fc1 = nn.Linear(65, 1)  # Corrected input size to match the LSTM output size
        self.dropout = nn.Dropout(0.78)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc1(x[:, -1, :]))
        return x
