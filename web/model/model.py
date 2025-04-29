from torch import nn, sigmoid

class Predictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size=1, kernel_size=3):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2,),
            nn.ReLU(),
        )
        
        self.memory_layer = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.3,
            num_layers=2,
            bidirectional=True,            
        )
        
        # Fully connected layers
        # The LSTM output size is doubled due to bidirectionality
        self.fc = nn.Sequential(  
            nn.Dropout(),
            nn.Linear(in_features=hidden_size*2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=output_size), 
        )
        
    def get_cnn_out_dim(self, in_dim: int, kernel_size: int, stride=1, padding=0, dilation=1): 
        return (in_dim + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
        
    def forward(self, x):
        output = self.cnn(x)
        
        lstm_input = output.permute(0, 2, 1)
        
        # Pass input through the LSTM
        out, _ = self.memory_layer(lstm_input)  # out shape: (batch_size, seq_len, hidden_size * 2)
        
        output = out[:, -1, :]  
        
        # Apply fully connected layers with ReLU activation
        output = self.fc(output)
        
        return sigmoid(output)