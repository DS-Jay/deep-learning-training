# Code Structure Overview for TensorFlow, PyTorch, and JAX

Deep learning frameworks like TensorFlow, PyTorch, and JAX provide distinct code structures and workflows for building, training, and deploying machine learning models. Understanding the code structure of each framework can help you choose the right tool for your projects. In this section, we'll explore the typical code structure for TensorFlow, PyTorch, and JAX, covering data preparation, model building, training, evaluation, and prediction.

### [Return to Main Page](../README.md)

#### TensorFlow (tf.keras)

**1. Setting Up the Environment:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

**2. Data Preparation:**
```python
# Example data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
```

**3. Model Building:**
```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

**4. Model Compilation:**
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**5. Model Training:**
```python
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

**6. Model Evaluation:**
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
```

**7. Model Prediction:**
```python
predictions = model.predict(X_test)
```

#### PyTorch

**1. Setting Up the Environment:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
```

**2. Data Preparation:**
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
```

**3. Model Building:**
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

model = Net()
```

**4. Model Compilation:**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**5. Model Training:**
```python
for epoch in range(5):
    for data in trainloader:
        X, y = data
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1} - Loss: {loss.item()}')
```

**6. Model Evaluation:**
```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        X, y = data
        output = model(X)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print(f'Accuracy: {round(correct/total, 3)}')
```

**7. Model Prediction:**
```python
with torch.no_grad():
    for data in testloader:
        X, y = data
        output = model(X)
        print(torch.argmax(output, dim=1))
        break
```

#### JAX

**1. Setting Up the Environment:**
```python
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
```

**2. Data Preparation:**
```python
# Example data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train.reshape(-1, 784), X_test.reshape(-1, 784)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)
```

**3. Model Building:**
```python
def relu(x):
    return jnp.maximum(0, x)

def predict(params, X):
    W1, b1, W2, b2, W3, b3 = params
    X = relu(jnp.dot(X, W1) + b1)
    X = relu(jnp.dot(X, W2) + b2)
    return jnp.dot(X, W3) + b3

def loss(params, X, y):
    preds = predict(params, X)
    return -jnp.mean(jnp.sum(y * preds, axis=1))
```

**4. Model Initialization:**
```python
key = random.PRNGKey(0)
input_size, hidden1, hidden2, output_size = 784, 128, 64, 10

W1 = random.normal(key, (input_size, hidden1))
b1 = jnp.zeros(hidden1)
W2 = random.normal(key, (hidden1, hidden2))
b2 = jnp.zeros(hidden2)
W3 = random.normal(key, (hidden2, output_size))
b3 = jnp.zeros(output_size)

params = [W1, b1, W2, b2, W3, b3]
```

**5. Model Compilation and Training:**
```python
learning_rate = 0.001

@jit
def update(params, X, y):
    grads = grad(loss)(params, X, y)
    return [(param - learning_rate * grad) for param, grad in zip(params, grads)]

for epoch in range(5):
    params = update(params, X_train, y_train)
    l = loss(params, X_train, y_train)
    print(f'Epoch {epoch+1}, Loss: {l}')
```

**6. Model Evaluation:**
```python
def accuracy(params, X, y):
    predictions = predict(params, X)
    return jnp.mean(jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1))

acc = accuracy(params, X_test, y_test)
print(f'Test accuracy: {acc}')
```

**7. Model Prediction:**
```python
predictions = predict(params, X_test)
print(jnp.argmax(predictions, axis=1))
```

### Summary

- **TensorFlow** uses a high-level API (Keras) for defining and training models with a focus on an integrated ecosystem for end-to-end workflows. Its code structure follows a sequential pattern of defining models, compiling, training, and evaluating.
  
- **PyTorch** is known for its dynamic computation graphs and user-friendly syntax, making it highly suitable for research and prototyping. The code structure involves defining models using classes, followed by training loops with manual gradient updates.
  
- **JAX** emphasizes a functional programming approach with an efficient automatic differentiation system. The code structure involves defining pure functions for predictions and loss calculations, with model parameters updated using JAX transformations like `grad` and `jit`.

Each framework has a distinct style and approach, making them suitable for different tasks, industries, and user experience levels.

### [Return to Main Page](../README.md)
# [TensorFlow Fundamental Concepts](04_tensorflow_fundamental_concepts.md)