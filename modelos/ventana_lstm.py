import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    # Dataset personalizado para secuencias de longitud variable
    def __init__(self, X_seqs, y_seqs):
        # Convertimos a tensores
        self.X_seqs = [torch.tensor(seq, dtype=torch.float32) for seq in X_seqs]
        self.y_seqs = torch.tensor(y_seqs, dtype=torch.long)

    # Obenemos la cantidad de secuencias en el dataset
    def __len__(self):
        return len(self.X_seqs)

    # Para obtener una secuencia y su etiqueta por indice
    def __getitem__(self, idx):
        return self.X_seqs[idx], self.y_seqs[idx]


# Funcion para hacer el padding y preparar los batches
def collate_fn(batch):
    # Separamos las secuencias y etiquetas
    X_batch, y_batch = zip(*batch)
    
    # Guardamos la longitud original de cada secuencia
    lengths = torch.tensor([len(seq) for seq in X_batch])
    
    # Hacemos el padding para que todas las secuencias tengan la misma longitud
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    
    # Convertimos las etiquetas a tensor
    y_batch = torch.tensor(y_batch)
    
    return X_padded, y_batch, lengths
