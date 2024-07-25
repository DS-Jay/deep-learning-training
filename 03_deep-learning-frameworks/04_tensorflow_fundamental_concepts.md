
# TensorFlow Fundamental Concepts

TensorFlow is a popular deep learning framework that provides a comprehensive ecosystem for building, training, and deploying machine learning models. In this section, we will explore fundamental concepts in TensorFlow, including model architecture, layers, optimizers, loss functions, and metrics.

### [Return to Main Page](../README.md)

### **1. Types of Normalization**

#### Batch Normalization
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Layer Normalization
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Instance Normalization
```python
import tensorflow_addons as tfa

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tfa.layers.InstanceNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Group Normalization
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tfa.layers.GroupNormalization(groups=4, axis=-1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Weight Normalization
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', use_bias=False),
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(64, activation='relu')),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### **2. Types of Models**

#### Feedforward Neural Network (FNN)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Convolutional Neural Network (CNN)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Recurrent Neural Network (RNN)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(50, input_shape=(100, 1)),
    tf.keras.layers.Dense(1, activation='linear')
])
```

#### Long Short-Term Memory Network (LSTM)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(100, 1)),
    tf.keras.layers.Dense(1, activation='linear')
])
```

#### Gated Recurrent Unit (GRU)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(50, input_shape=(100, 1)),
    tf.keras.layers.Dense(1, activation='linear')
])
```

#### Transformer
```python
from tensorflow.keras.layers import MultiHeadAttention

inputs = tf.keras.Input(shape=(None, 512))
attention = MultiHeadAttention(num_heads=8, key_dim=512)(inputs, inputs)
model = tf.keras.Model(inputs=inputs, outputs=attention)
```

#### Autoencoder
```python
encoding_dim = 32
input_img = tf.keras.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_img, decoded)
```

#### Generative Adversarial Network (GAN)
```python
# Generator
generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(784, activation='sigmoid')
])

# Discriminator
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### **3. Architectures**

#### Single Input
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Multi-Input
```python
input1 = tf.keras.Input(shape=(28, 28, 1))
input2 = tf.keras.Input(shape=(10,))
x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input1)
x1 = tf.keras.layers.Flatten()(x1)
x2 = tf.keras.layers.Dense(10, activation='relu')(input2)
concat = tf.keras.layers.concatenate([x1, x2])
output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
model = tf.keras.Model(inputs=[input1, input2], outputs=output)
```

#### Single Output
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
```

#### Multi-Output
```python
input = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input)
x = tf.keras.layers.Flatten()(x)
output1 = tf.keras.layers.Dense(10, activation='softmax')(x)
output2 = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=input, outputs=[output1, output2])
```

#### Encoder-Decoder
```python
# Encoder
encoder_input = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(encoder_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
encoder_output = tf.keras.layers.Flatten()(x)
encoder = tf.keras.Model(encoder_input, encoder_output)

# Decoder
decoder_input = tf.keras.Input(shape=(3136,))
x = tf.keras.layers.Reshape((7, 7, 64))(decoder_input)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid')(x)
decoder = tf.keras.Model(decoder_input, decoder_output)

# Autoencoder
autoencoder_input = tf.keras.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = tf.keras.Model(autoencoder_input, decoded)
```

#### Attention Mechanisms
```python
inputs = tf.keras.Input(shape=(None, 512))
attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=512)(inputs, inputs)
model = tf.keras.Model(inputs=inputs, outputs=attention)
```

### **4. Layers and Their Use Cases**

#### Dense Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Convolutional Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Pooling Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Recurrent Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(50, input_shape=(100, 1)),
    tf.keras.layers.Dense(1, activation='linear')
])
```

#### Dropout Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28, 28)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Batch Normalization Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28, 28)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Embedding Layer
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=10),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### **5. Optimizers**

```python
# Stochastic Gradient

 Descent (SGD)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# RMSProp
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# AdaGrad
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

# Adadelta
optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)
```

### **6. Loss Functions**

```python
# Mean Squared Error (MSE)
loss = 'mean_squared_error'

# Mean Absolute Error (MAE)
loss = 'mean_absolute_error'

# Binary Cross-Entropy
loss = 'binary_crossentropy'

# Categorical Cross-Entropy
loss = 'categorical_crossentropy'

# Sparse Categorical Cross-Entropy
loss = 'sparse_categorical_crossentropy'

# Huber Loss
loss = 'huber'
```

### **7. Metrics**

```python
metrics = ['accuracy']

# Precision
metrics = [tf.keras.metrics.Precision()]

# Recall
metrics = [tf.keras.metrics.Recall()]

# F1 Score
metrics = [tf.keras.metrics.AUC(curve='PR')]

# AUC-ROC
metrics = [tf.keras.metrics.AUC()]
```

### **8. Model Compilation**

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### **9. Model Fitting**

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

### **10. Model Evaluation**

```python
model.evaluate(x_test, y_test)
```

### **11. Activation Functions**

```python
# ReLU (Rectified Linear Unit)
activation = 'relu'

# Sigmoid
activation = 'sigmoid'

# Tanh
activation = 'tanh'

# Softmax
activation = 'softmax'

# Leaky ReLU
activation = tf.keras.layers.LeakyReLU(alpha=0.01)
```

### **12. List of Parameters**

```python
# Weights and Biases are automatically learned during training

# Learning Rate
learning_rate = 0.001

# Batch Size
batch_size = 32

# Epochs
epochs = 5
```

---

### [Return to Main Page](../README.md)
# [TensorFlow Notebook](tensorflow_notebook.ipynb)