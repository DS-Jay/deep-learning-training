 
## PyTorch Fundamental Concepts

PyTorch is a popular deep learning framework known for its dynamic computation graphs and user-friendly syntax. This section covers fundamental concepts in PyTorch, including normalization techniques, different types of models and architectures, layers and their use cases, optimizers, loss functions, metrics, model compilation, fitting, evaluation, activation functions, and parameters.

### [Return to Main Page](../README.md)

### **1. Types of Normalization**

#### Batch Normalization
```python
import torch
import torch.nn as nn

class BatchNormModel(nn.Module):
    def __init__(self):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = BatchNormModel()
```

#### Layer Normalization
```python
class LayerNormModel(nn.Module):
    def __init__(self):
        super(LayerNormModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = LayerNormModel()
```

#### Instance Normalization
```python
class InstanceNormModel(nn.Module):
    def __init__(self):
        super(InstanceNormModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.in1(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.softmax(self.fc1(x), dim=1)
        return x

model = InstanceNormModel()
```

#### Group Normalization
```python
class GroupNormModel(nn.Module):
    def __init__(self):
        super(GroupNormModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 64)
        self.fc1 = nn.Linear(64 * 28 * 28, 10)

    def forward(self, x):
        x = torch.relu(self.gn1(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.softmax(self.fc1(x), dim=1)
        return x

model = GroupNormModel()
```

#### Weight Normalization
```python
import torch.nn.utils as nn_utils

class WeightNormModel(nn.Module):
    def __init__(self):
        super(WeightNormModel, self).__init__()
        self.fc1 = nn_utils.weight_norm(nn.Linear(784, 64))
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = WeightNormModel()
```

### **2. Types of Models**

#### Feedforward Neural Network (FNN)
```python
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = FNN()
```

#### Convolutional Neural Network (CNN)
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = CNN()
```

#### Recurrent Neural Network (RNN)
```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
output_size = 1
model = RNN(input_size, hidden_size, output_size)
```

#### Long Short-Term Memory Network (LSTM)
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
output_size = 1
model = LSTM(input_size, hidden_size, output_size)
```

#### Gated Recurrent Unit (GRU)
```python
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
output_size = 1
model = GRU(input_size, hidden_size, output_size)
```

#### Transformer
```python
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = F.relu(self.fc(x.mean(dim=1)))
        return x

input_dim = 512
num_heads = 8
num_layers = 6
model = Transformer(input_dim, num_heads, num_layers)
```

#### Autoencoder
```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
```

#### Generative Adversarial Network (GAN)
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

generator = Generator()
discriminator = Discriminator

()
```

### **3. Architectures**

#### Single Input
```python
class SingleInputModel(nn.Module):
    def __init__(self):
        super(SingleInputModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = SingleInputModel()
```

#### Multi-Input
```python
class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim=1)
        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc2(x2))
        x = torch.cat((x1, x2), dim=1)
        x = torch.softmax(self.fc3(x), dim=1)
        return x

model = MultiInputModel()
```

#### Single Output
```python
class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SingleOutputModel()
```

#### Multi-Output
```python
class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        output1 = torch.softmax(self.fc2(x), dim=1)
        output2 = self.fc3(x)
        return output1, output2

model = MultiOutputModel()
```

#### Encoder-Decoder
```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample(x)
        x = torch.sigmoid(self.conv1(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
```

#### Attention Mechanisms
```python
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(AttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        output = F.relu(self.fc(attn_output.mean(dim=1)))
        return output

input_dim = 512
num_heads = 8
model = AttentionModel(input_dim, num_heads)
```

### **4. Layers and Their Use Cases**

#### Dense Layer
```python
class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = DenseModel()
```

#### Convolutional Layer
```python
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = ConvModel()
```

#### Pooling Layer
```python
class PoolingModel(nn.Module):
    def __init__(self):
        super(PoolingModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = PoolingModel()
```

#### Recurrent Layer
```python
class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
output_size = 1
model = RecurrentModel(input_size, hidden_size, output_size)
```

#### Dropout Layer
```python
class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = DropoutModel()
```

#### Batch Normalization Layer
```python
class BatchNormModel(nn.Module):
    def __init__(self):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = BatchNormModel()
```

#### Embedding Layer
```python
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(1000, 64)
        self.fc1 = nn.Linear(64 * 10, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.softmax(self.fc1(x), dim=1)
        return x

model = EmbeddingModel()
```

### **5. Optimizers**

```python
# Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = torch

.optim.Adam(model.parameters(), lr=0.001)

# RMSProp
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# AdaGrad
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

# Adadelta
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
```

### **6. Loss Functions**

```python
# Mean Squared Error (MSE)
loss_fn = nn.MSELoss()

# Mean Absolute Error (MAE)
loss_fn = nn.L1Loss()

# Binary Cross-Entropy
loss_fn = nn.BCELoss()

# Categorical Cross-Entropy
loss_fn = nn.CrossEntropyLoss()

# Sparse Categorical Cross-Entropy
loss_fn = nn.CrossEntropyLoss()

# Huber Loss
loss_fn = nn.SmoothL1Loss()
```

### **7. Metrics**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred):
    y_pred = y_pred.argmax(dim=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc_roc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    return accuracy, precision, recall, f1, auc_roc
```

### **8. Model Compilation**

PyTorch does not have a direct `compile` method like TensorFlow. Instead, you define the loss function and optimizer separately.

```python
model = FNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### **9. Model Fitting**

```python
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
```

### **10. Model Evaluation (continued)**

```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        y_pred = model(x_batch)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
```

### **11. Activation Functions**

```python
import torch.nn.functional as F

# ReLU (Rectified Linear Unit)
activation = F.relu

# Sigmoid
activation = torch.sigmoid

# Tanh
activation = torch.tanh

# Softmax
activation = F.softmax

# Leaky ReLU
activation = F.leaky_relu
```

### **12. List of Parameters**

```python
# Weights and Biases are automatically learned during training
# No need to manually define them as they are part of the model's layers

# Learning Rate
learning_rate = 0.001

# Batch Size
batch_size = 32

# Epochs
epochs = 5
```

### **Complete Example of Training a Model**

Here is a complete example of training a simple feedforward neural network on the MNIST dataset using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = FeedforwardNN()

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        y_pred = model(x_batch)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
```


### [Return to Main Page](../README.md)

# [Pytorch Notebook](pytorch_notebook.ipynb)    