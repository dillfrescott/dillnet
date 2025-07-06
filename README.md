# dillnet

Dill custom model.

### Usage:
```python
import torch
from dillnet import DillNet

# Define model configuration
dim = 512         # Dimensionality
depth = 6         # Number of layers
heads = 8         # Number of heads

# Initialize the DillNet model
model = DillNet(dim=dim, depth=depth, heads=heads)

# Create a sample input (batch of token embeddings)
# For example: batch size = 2, sequence length = 128
x = torch.randn(2, 128, dim)

# Pass input through the model
output = model(x)

# Output shape should match input: (batch_size, seq_length, dim)
print("Output shape:", output.shape)
```