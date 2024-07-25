
# JAX Fundamental Concepts

JAX is a numerical computing library that provides composable function transformations and efficient GPU/TPU support. It is designed for high-performance machine learning research and is widely used for deep learning applications. In this section, we will explore fundamental concepts in JAX, including tensors, automatic differentiation, and neural network building blocks.

### [Return to Main Page](../README.md)

### **1. Types of Normalization**

In JAX, normalization layers can be implemented using the Haiku or Flax libraries.

#### Batch Normalization
```python
import jax.numpy as jnp
import haiku as hk

class BatchNormModel(hk.Module):
    def __call__(self, x, is_training):
        x = hk.Linear(64)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x, is_training):
    model = BatchNormModel()
    return model(x, is_training)

model = hk.transform_with_state(forward_fn)
```

#### Layer Normalization
```python
class LayerNormModel(hk.Module):
    def __call__(self, x):
        x = hk.Linear(64)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = LayerNormModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Instance Normalization
```python
class InstanceNormModel(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(64, kernel_shape=3, padding='SAME')(x)
        x = hk.InstanceNorm(create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = InstanceNormModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Group Normalization
```python
class GroupNormModel(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(64, kernel_shape=3, padding='SAME')(x)
        x = hk.GroupNorm(groups=4, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = GroupNormModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Weight Normalization
```python
class WeightNormModel(hk.Module):
    def __call__(self, x):
        x = hk.Linear(64, w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "normal"))(x)
        x = hk.WeightNorm()(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = WeightNormModel()
    return model(x)

model = hk.transform(forward_fn)
```

### **2. Types of Models**

#### Feedforward Neural Network (FNN)
```python
class FNN(hk.Module):
    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = FNN()
    return model(x)

model = hk.transform(forward_fn)
```

#### Convolutional Neural Network (CNN)
```python
class CNN(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(32, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = CNN()
    return model(x)

model = hk.transform(forward_fn)
```

#### Recurrent Neural Network (RNN)
```python
class RNN(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, x, prev_h):
        x = hk.Linear(self.hidden_size)(x)
        h = jax.nn.tanh(x + prev_h)
        return h, h

def forward_fn(x, prev_h):
    model = RNN(50)
    return model(x, prev_h)

model = hk.transform(forward_fn)
```

#### Long Short-Term Memory Network (LSTM)
```python
class LSTM(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = hk.LSTM(hidden_size)

    def __call__(self, x, prev_state):
        return self.lstm(x, prev_state)

def forward_fn(x, prev_state):
    model = LSTM(50)
    return model(x, prev_state)

model = hk.transform(forward_fn)
```

#### Gated Recurrent Unit (GRU)
```python
class GRU(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = hk.GRU(hidden_size)

    def __call__(self, x, prev_state):
        return self.gru(x, prev_state)

def forward_fn(x, prev_state):
    model = GRU(50)
    return model(x, prev_state)

model = hk.transform(forward_fn)
```

#### Transformer
```python
class Transformer(hk.Module):
    def __init__(self, num_heads, num_layers):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers

    def __call__(self, x):
        for _ in range(self.num_layers):
            x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=64)(x, x, x)
            x = jax.nn.relu(x)
        x = hk.Linear(1)(x)
        return x

def forward_fn(x):
    model = Transformer(8, 6)
    return model(x)

model = hk.transform(forward_fn)
```

#### Autoencoder
```python
class Autoencoder(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        encoded = hk.Linear(32)(x)
        x = jax.nn.relu(encoded)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        decoded = hk.Linear(784)(x)
        return decoded

def forward_fn(x):
    model = Autoencoder()
    return model(x)

model = hk.transform(forward_fn)
```

#### Generative Adversarial Network (GAN)
```python
class Generator(hk.Module):
    def __call__(self, z):
        x = hk.Linear(128)(z)
        x = jax.nn.relu(x)
        x = hk.Linear(784)(x)
        return jax.nn.tanh(x)

class Discriminator(hk.Module):
    def __call__(self, x):
        x = hk.Linear(128)(x)
        x = jax.nn.leaky_relu(x, 0.2)
        x = hk.Linear(1)(x)
        return jax.nn.sigmoid(x)

def generator_fn(z):
    model = Generator()
    return model(z)

def discriminator_fn(x):
    model = Discriminator()
    return model(x)

generator = hk.transform(generator_fn)
discriminator = hk.transform(discriminator_fn)
```

### **3. Architectures**

#### Single Input
```python
class SingleInputModel(hk.Module):
    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = SingleInputModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Multi-Input
```python
class MultiInputModel(hk.Module):
    def __call__(self, x1, x2):
        x1 = hk.Flatten()(x1)
        x1 = hk.Linear(128)(x1)
        x1 = jax.nn.relu(x1)
        x2 = hk.Linear(10)(x2)
        x2 = jax.nn.relu(x2)
        x = jnp.concatenate([x1, x2], axis=-1)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x1, x2):
    model = MultiInputModel()
    return model(x1, x2)

model = hk.transform(forward_fn)
```

#### Single Output
```python
class SingleOutputModel(hk.Module):
    def __call__(self, x):
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(1)(x)
        return x

def forward_fn(x):


    model = SingleOutputModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Multi-Output
```python
class MultiOutputModel(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(32, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        output1 = hk.Linear(10)(x)
        output2 = hk.Linear(1)(x)
        return output1, output2

def forward_fn(x):
    model = MultiOutputModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Encoder-Decoder
```python
class Encoder(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(64, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)
        return x

class Decoder(hk.Module):
    def __call__(self, x):
        x = hk.Conv2DTranspose(64, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.UpSample(scale_factor=2)(x)
        x = hk.Conv2DTranspose(1, kernel_shape=3, padding='SAME')(x)
        return jax.nn.sigmoid(x)

class Autoencoder(hk.Module):
    def __call__(self, x):
        encoder = Encoder()
        decoder = Decoder()
        x = encoder(x)
        x = decoder(x)
        return x

def forward_fn(x):
    model = Autoencoder()
    return model(x)

model = hk.transform(forward_fn)
```

#### Attention Mechanisms
```python
class AttentionModel(hk.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def __call__(self, x):
        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=64)(x, x, x)
        x = jax.nn.relu(x)
        x = hk.Linear(1)(x)
        return x

def forward_fn(x):
    model = AttentionModel(8)
    return model(x)

model = hk.transform(forward_fn)
```

### **4. Layers and Their Use Cases**

#### Dense Layer
```python
class DenseModel(hk.Module):
    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = DenseModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Convolutional Layer
```python
class ConvModel(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(32, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = ConvModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Pooling Layer
```python
class PoolingModel(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(32, kernel_shape=3, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = PoolingModel()
    return model(x)

model = hk.transform(forward_fn)
```

#### Recurrent Layer
```python
class RecurrentModel(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = hk.RNN(hidden_size)

    def __call__(self, x, prev_state):
        return self.rnn(x, prev_state)

def forward_fn(x, prev_state):
    model = RecurrentModel(50)
    return model(x, prev_state)

model = hk.transform(forward_fn)
```

#### Dropout Layer
```python
class DropoutModel(hk.Module):
    def __call__(self, x, is_training):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), 0.5, x) if is_training else x
        x = hk.Linear(10)(x)
        return x

def forward_fn(x, is_training):
    model = DropoutModel()
    return model(x, is_training)

model = hk.transform(forward_fn)
```

#### Batch Normalization Layer
```python
class BatchNormModel(hk.Module):
    def __call__(self, x, is_training):
        x = hk.Linear(64)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x, is_training):
    model = BatchNormModel()
    return model(x, is_training)

model = hk.transform_with_state(forward_fn)
```

#### Embedding Layer
```python
class EmbeddingModel(hk.Module):
    def __call__(self, x):
        x = hk.Embed(vocab_size=1000, embed_dim=64)(x)
        x = hk.Flatten()(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = EmbeddingModel()
    return model(x)

model = hk.transform(forward_fn)
```

### **5. Optimizers**

In JAX, optimizers can be implemented using the Optax library.

```python
import optax

# Stochastic Gradient Descent (SGD)
optimizer = optax.sgd(learning_rate=0.01)

# Momentum
optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# Adam
optimizer = optax.adam(learning_rate=0.001)

# RMSProp
optimizer = optax.rmsprop(learning_rate=0.001)

# AdaGrad
optimizer = optax.adagrad(learning_rate=0.01)

# Adadelta
optimizer = optax.adadelta(learning_rate=1.0)
```

### **6. Loss Functions**

```python
import jax.numpy as jnp

# Mean Squared Error (MSE)
def mse_loss(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

# Mean Absolute Error (MAE)
def mae_loss(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))

# Binary Cross-Entropy
def binary_crossentropy_loss(y_true, y_pred):
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

# Categorical Cross-Entropy
def categorical_crossentropy_loss(y_true, y_pred):
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))

# Sparse Categorical Cross-Entropy
def sparse_categorical_crossentropy_loss(y_true, y_pred):
    return -jnp.mean(jnp.log(y_pred[jnp.arange(len(y_true)), y_true]))

# Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = jnp.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (jnp.abs(error) - 0.5 * delta)
    return jnp.mean(jnp.where(is_small_error, squared_loss, linear_loss))
```

### **7. Metrics**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_metrics(y_true, y_pred):
    y_pred = jnp.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc_roc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    return accuracy, precision, recall, f1, auc_roc
```

### **8. Model Compilation**

In JAX, there isn't a direct `compile` method like in TensorFlow.

 You define the loss function and optimizer separately and manage training loops manually.

### **9. Model Fitting**

```python
from jax import jit, grad
import jax

@jit
def update(params, x, y, opt_state):
    def loss_fn(params):
        y_pred = model.apply(params, None, x)
        loss = categorical_crossentropy_loss(y, y_pred)
        return loss

    grads = grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

params = model.init(jax.random.PRNGKey(0), jnp.ones([32, 28, 28, 1]))
opt_state = optimizer.init(params)

num_epochs = 5
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        params, opt_state = update(params, x_batch, y_batch, opt_state)
```

### **10. Model Evaluation**

```python
@jit
def evaluate(params, x):
    y_pred = model.apply(params, None, x)
    return y_pred

model.eval()
with jax.disable_jit():
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        y_pred = evaluate(params, x_batch)
        predicted = jnp.argmax(y_pred, axis=1)
        total += y_batch.shape[0]
        correct += jnp.sum(predicted == y_batch)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
```

### **11. Activation Functions**

```python
# ReLU (Rectified Linear Unit)
activation = jax.nn.relu

# Sigmoid
activation = jax.nn.sigmoid

# Tanh
activation = jax.nn.tanh

# Softmax
activation = jax.nn.softmax

# Leaky ReLU
def leaky_relu(x, negative_slope=0.01):
    return jnp.where(x > 0, x, negative_slope * x)
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

Here is a complete example of training a simple feedforward neural network on the MNIST dataset using JAX and Haiku:

```python
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax import grad, jit
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Loading and Preprocessing
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

batch_size = 32
train_loader = jax.dlpack.from_dlpack_iter((X_train[i:i + batch_size], y_train[i:i + batch_size]) for i in range(0, len(X_train), batch_size))
test_loader = jax.dlpack.from_dlpack_iter((X_test[i:i + batch_size], y_test[i:i + batch_size]) for i in range(0, len(X_test), batch_size))

# Model Definition
class FeedforwardNN(hk.Module):
    def __call__(self, x):
        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x

def forward_fn(x):
    model = FeedforwardNN()
    return model(x)

model = hk.transform(forward_fn)

# Loss Function and Optimizer
def loss_fn(params, x, y):
    y_pred = model.apply(params, None, x)
    return categorical_crossentropy_loss(y, y_pred)

optimizer = optax.adam(learning_rate=0.001)

# Training Loop
@jit
def update(params, opt_state, x, y):
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

params = model.init(jax.random.PRNGKey(0), jnp.ones([batch_size, 28 * 28]))
opt_state = optimizer.init(params)

num_epochs = 5
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        params, opt_state = update(params, opt_state, x_batch, y_batch)
    print(f'Epoch {epoch + 1}/{num_epochs} completed.')

# Evaluation
model.eval()
with jax.disable_jit():
    correct = 0
    total = 0
    for x_batch, y_batch in test_loader:
        y_pred = model.apply(params, None, x_batch)
        predicted = jnp.argmax(y_pred, axis=1)
        total += y_batch.shape[0]
        correct += jnp.sum(predicted == y_batch)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')
```

### [Return to Main Page](../README.md)

# [JAX Notebook](jax_deep_learning.ipynb)