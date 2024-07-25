# Prerequisites and Setup

### [Return to Main Page](../README.md)

## Prerequisites
Before you start this course, make sure you have a basic understanding of the following topics:
- Python programming
- Linear algebra
- Calculus
- Probability and statistics

## Environment Setup
To get started with deep learning, you'll need to set up your environment. We'll guide you through the process of installing TensorFlow, PyTorch, and JAX, and setting up Jupyter Notebook.

### Step 1: Install Anaconda
We recommend using Anaconda to manage your Python environment and packages. Download and install Anaconda from [here](https://www.anaconda.com/products/individual).

### Step 2: Create a New Conda Environment
Create a new environment for this course with Python 3.11:
```bash
conda create -n deep-learning-course python=3.11
conda activate deep-learning-course
```

Step 3: Install Jupyter Notebook
Install Jupyter Notebook in your new environment:
```bash
conda install jupyter
```

Step 4: Install TensorFlow

Any issues reference the official [TensorFlow](https://www.tensorflow.org/install) installation guide.

Install TensorFlow:
```bash
conda install tensorflow
```

Step 5: Install PyTorch

Any issues reference the official [PyTorch](https://pytorch.org/get-started/locally/) installation guide.

Install PyTorch.  
```bash
pip install torch torchvision torchaudio


```

Step 6: Install JAX

Any issues reference the official [JAX](https://jax.readthedocs.io/en/latest/installation.html) installation guide.

Install JAX:
```bash
pip install jax jaxlib
```

Step 7: Verify the Installation
Open Jupyter Notebook and create a new notebook to verify the installation of TensorFlow, PyTorch, and JAX. Run the following commands in separate cells to ensure everything is installed correctly:
```python
import tensorflow as tf
print(tf.__version__)

import torch
print(torch.__version__)

import jax
print(jax.__version__)
```
If you see the version numbers printed without errors, your setup is complete!


### [Return to Main Page](../README.md)
### [Notebook Setup](setup_notebook.ipynb)
### [Next: Formulas](../02_fundamental-concepts/formulas.md)

