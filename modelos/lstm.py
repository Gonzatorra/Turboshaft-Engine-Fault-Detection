import torch
import torch.nn as nn

class LSTM_basico(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=2, dropout=0.3):
        """
        Arquitectura LSTM para clasificación multiclase.

        Args:
            input_dim (int): número de características por paso de tiempo.
            hidden_dim (int): número de neuronas en cada capa LSTM.
            num_layers (int): número de capas LSTM apiladas.
            output_dim (int): número de clases a predecir.
            dropout (float): dropout entre capas LSTM para regularización.
        """
        super(LSTM_basico, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)              # salida de LSTM: (batch, seq_len, hidden_dim)
        out = lstm_out[:, -1, :]                # tomar la salida del último paso de tiempo
        out = self.fc(out)                      # capa final para clasificación
        return out
