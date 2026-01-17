import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BiLSTMClassifier(L.LightningModule):
    """Model BiLSTM do klasyfikacji hate speech"""

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 output_dim: int, dropout1: float = 0.6,
                 dropout2: float = 0.3):
        super().__init__()

        # Warstwy
        self.conv1d = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding='same')
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(32, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim // 2)
        self.fc3 = nn.Linear(output_dim // 2, 1)

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        self.criterion = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        x = x.permute(0, 2, 1)  # (batch, emb_dim, seq_len)

        # Conv1D
        x = F.relu(self.conv1d(x))
        x = self.maxpool(x)

        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # FC layers
        x = torch.sigmoid(self.fc1(hidden))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        output = self.fc3(x)

        return output

    def training_step(self, batch, batch_idx):
        text, labels = batch
        logits = self(text).squeeze(1)
        loss = self.criterion(logits, labels.float())

        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        text, labels = batch
        logits = self(text).squeeze(1)
        loss = self.criterion(logits, labels.float())

        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        text, labels = batch
        logits = self(text).squeeze(1)
        loss = self.criterion(logits, labels.float())

        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }