# Fundamental Concepts

### [Return to Main Page](../README.md)

## Neural Network Basics

### Introduction to Artificial Neural Networks (ANN)
Artificial Neural Networks (ANNs) are computational models inspired by the human brain. They consist of interconnected nodes, or neurons, organized in layers. Each neuron performs a simple computation and passes the result to the next layer of neurons.

### How Does an ANN Work?

1. **Input Layer**: The input layer receives raw data. Each neuron in this layer corresponds to a feature in the input data. For example, in a dataset of house prices, features could include the number of bedrooms, square footage, and location.

2. **Hidden Layers**: These layers perform computations to extract patterns and features from the input data. Each neuron in a hidden layer takes a weighted sum of the inputs, applies an activation function, and passes the result to the next layer. The activation function introduces non-linearity, allowing the network to learn complex patterns. 

3. **Output Layer**: The final layer produces the output of the network. This could be a single value, a set of values, or a probability distribution. For instance, in a house price prediction model, the output layer might produce a single value representing the predicted price.

### Use Case: Predicting House Prices

Consider a business scenario where you want to predict house prices based on various features (e.g., number of bedrooms, square footage, location). Here's how an ANN would work:

- **Input Layer**: Receives input features such as number of bedrooms, square footage, and location.
- **Hidden Layers**: Process these features through a series of weighted sums and activation functions to learn complex patterns, like the interaction between square footage and location.
- **Output Layer**: Produces the predicted house price based on the learned patterns.

### Comparison with Traditional Machine Learning Algorithms

- **Feature Extraction**: Traditional algorithms often require manual feature extraction and selection. ANNs automatically learn features during training.
- **Complexity**: ANNs can model complex, non-linear relationships, making them suitable for tasks like image recognition and natural language processing.
- **Scalability**: ANNs can handle large-scale data and perform well with high-dimensional data.

## Activation Functions

### Purpose of Activation Functions
Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.

### Common Activation Functions

#### Sigmoid
- **Mathematical Definition**: 
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- **Explanation**: The sigmoid function maps any real-valued number into the range (0, 1). It's useful for models where we need to predict probabilities.
- **Pros**: Smooth gradient, output values bound between 0 and 1.
- **Cons**: Vanishing gradient problem, outputs are not zero-centered.
- **Use Cases**: Binary classification problems.

#### Tanh
- **Mathematical Definition**: 
  $$\tanh(x) = \frac{2}{1 + e^{-2x}} - 1$$
- **Explanation**: The tanh function maps any real-valued number into the range (-1, 1). It is a scaled sigmoid function.
- **Pros**: Outputs are zero-centered, smoother gradients than sigmoid.
- **Cons**: Vanishing gradient problem.
- **Use Cases**: Usually preferred over sigmoid for hidden layers.

#### ReLU (Rectified Linear Unit)
- **Mathematical Definition**: 
  $$f(x) = \max(0, x)$$
- **Explanation**: The ReLU function outputs the input directly if it is positive; otherwise, it will output zero.
- **Pros**: Computationally efficient, converges faster.
- **Cons**: Can cause dead neurons during training (neurons that never activate).
- **Use Cases**: Most commonly used activation function in hidden layers.

#### Leaky ReLU
- **Mathematical Definition**: 
  $$f(x) = \max(\alpha x, x)$$ 
  where alpha is a small constant (e.g., 0.01).
- **Explanation**: A variant of ReLU that allows a small, non-zero gradient when the unit is not active.
- **Pros**: Helps mitigate the dying ReLU problem.
- **Cons**: Introduces a slight computational overhead.
- **Use Cases**: An improvement over ReLU for some models.

#### Softmax
- **Mathematical Definition**: 
  $$\sigma(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
- **Explanation**: The softmax function converts a vector of values into a probability distribution. Each value is mapped to the range (0, 1), and the sum of all output values is 1.
- **Pros**: Useful for multi-class classification.
- **Cons**: Computationally intensive.
- **Use Cases**: Output layer of classification models.

## Loss Functions

### Understanding Loss and Cost Functions
Loss functions measure the difference between the predicted output and the actual output. They guide the optimization process by providing feedback.

### Common Loss Functions

#### Mean Squared Error (MSE)
- **Mathematical Definition**: 
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Explanation**: Measures the average squared difference between the actual and predicted values.
- **Use Cases**: Regression problems.

#### Cross-Entropy Loss
- **Mathematical Definition**: 
  $$\text{Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} y_i \log(\hat{y}_i)$$
- **Explanation**: Measures the performance of a classification model whose output is a probability value between 0 and 1.
- **Use Cases**: Binary and multi-class classification problems.

#### Hinge Loss
- **Mathematical Definition**: 
  $$\text{Hinge} = \max(0, 1 - y_i \hat{y}_i)$$
- **Explanation**: Used for "maximum-margin" classification, most notably for support vector machines.
- **Use Cases**: Binary classification problems with SVMs.

## Hands-on Examples
For hands-on examples of these concepts, please refer to the corresponding Jupyter notebook.

[Go to Concepts Notebook](concepts_notebook.ipynb)

### [Return to Main Page](../README.md)
### [Next: Fundamental Concepts](../02_fundamental-concepts/01_fundamental_concepts_dl.md)