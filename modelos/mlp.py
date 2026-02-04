import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        #3 capas densas
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x