# eeg_processor.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EEGModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.dropout(last_output)
        x = self.fc(x)
        return x

class EEGProcessor:
    def __init__(self, num_channels=8, sequence_length=100):
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.model = None
        self.vocabulary = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_signal(self, signal):
        """Preprocess EEG signal"""
        # Normalize
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # Apply bandpass filter (simulate 0.5-50Hz)
        from scipy.signal import butter, filtfilt
        nyq = 256 / 2
        b, a = butter(4, [0.5/nyq, 50/nyq], btype='band')
        filtered_signal = np.zeros_like(signal)
        for channel in range(signal.shape[-1]):
            filtered_signal[:, channel] = filtfilt(b, a, signal[:, channel])
            
        return filtered_signal
    
    def build_model(self, num_classes):
        """Build the PyTorch model"""
        self.model = EEGModel(
            input_size=self.num_channels,
            hidden_size=64,
            num_classes=num_classes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Preprocess all training data
        X_processed = np.array([self.preprocess_signal(x) for x in X])
        
        # Split data
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        # Create datasets
        train_dataset = EEGDataset(X_processed[train_idx], y[train_idx])
        val_dataset = EEGDataset(X_processed[val_idx], y[val_idx])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Build model if not exists
        if self.model is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_correct = 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
            
            # Validation
            self.model.eval()
            val_loss, val_correct = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
            
            # Record history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_correct / len(train_dataset))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_correct / len(val_dataset))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {history["train_loss"][-1]:.4f}, '
                      f'Train Acc: {history["train_acc"][-1]:.4f}')
                print(f'Val Loss: {history["val_loss"][-1]:.4f}, '
                      f'Val Acc: {history["val_acc"][-1]:.4f}')
        
        return history
    
    def predict(self, signal, return_confidence=False):
        """Predict word from EEG signal"""
        self.model.eval()
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal)
        processed_signal = torch.FloatTensor(processed_signal).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(processed_signal)
            probabilities = torch.softmax(outputs, dim=1)
            word_idx = outputs.argmax(1).item()
            confidence = probabilities[0, word_idx].item()
        
        if return_confidence:
            return word_idx, confidence
        return word_idx
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(filepath))
        
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()