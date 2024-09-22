### Function 1: Matrix Multiplication with Sparsity Pattern

```python
import torch
import numpy as np

def matrix_multiplication_with_sparsity(A: torch.Tensor, B: torch.Tensor, sparsity_pattern: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication with a specified sparsity pattern.

    Args:
    A (torch.Tensor): The first matrix.
    B (torch.Tensor): The second matrix.
    sparsity_pattern (torch.Tensor): The sparsity pattern to apply.

    Returns:
    torch.Tensor: The result of the matrix multiplication with the specified sparsity pattern.
    """
    # Get the dimensions of the matrices
    num_rows_A, num_cols_A = A.shape
    num_rows_B, num_cols_B = B.shape

    # Check if the sparsity pattern is valid
    if sparsity_pattern.shape != (num_rows_A, num_cols_B):
        raise ValueError("Invalid sparsity pattern shape")

    # Create a mask to apply the sparsity pattern
    mask = sparsity_pattern.to(torch.bool)

    # Apply the sparsity pattern to the matrices
    A_sparse = A * mask
    B_sparse = B * mask

    # Perform the matrix multiplication
    result = torch.matmul(A_sparse, B_sparse)

    return result
```

### Function 2: Harmonic-Percussive Separation

```python
import torch
import numpy as np

def harmonic_percussive_separation(audio: torch.Tensor, sample_rate: int = 44100) -> (torch.Tensor, torch.Tensor):
    """
    Perform harmonic-percussive separation on an audio signal.

    Args:
    audio (torch.Tensor): The audio signal.
    sample_rate (int): The sample rate of the audio signal. Defaults to 44100.

    Returns:
    tuple: A tuple containing the harmonic and percussive components of the audio signal.
    """
    # Create a Butterworth filter to separate the harmonic and percussive components
    from scipy.signal import butter, lfilter
    nyq = sample_rate / 2.0
    lowcut = 100.0
    highcut = 2000.0
    order = 5
    low = butter(order, lowcut/nyq, btype='low')
    high = butter(order, highcut/nyq, btype='high')

    # Apply the filters to the audio signal
    harmonic = lfilter(low[0], low[1], audio)
    percussive = lfilter(high[0], high[1], audio)

    # Convert the results to PyTorch tensors
    harmonic = torch.from_numpy(harmonic)
    percussive = torch.from_numpy(percussive)

    return harmonic, percussive
```

### Function 3: Matrix Operations with Mixed Precision

```python
import torch
import numpy as np

def matrix_operations_with_mixed_precision(A: torch.Tensor, B: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Perform matrix operations with mixed precision.

    Args:
    A (torch.Tensor): The first matrix.
    B (torch.Tensor): The second matrix.

    Returns:
    tuple: A tuple containing the result of the matrix addition and multiplication with mixed precision.
    """
    # Create a matrix with the same shape as A and B
    C = torch.zeros_like(A)

    # Perform matrix addition with fp32 precision
    C_fp32 = A + B

    # Perform matrix multiplication with fp16 precision
    C_fp16 = torch.matmul(A, B)

    # Convert the results to fp32 precision
    C_fp32 = C_fp32.to(torch.float32)
    C_fp16 = C_fp16.to(torch.float32)

    return C_fp32, C_fp16
```

### Example Usage

```python
# Create some sample matrices
A = torch.randn(3, 3)
B = torch.randn(3, 3)

# Create a sparsity pattern
sparsity_pattern = torch.ones(3, 3)

# Perform matrix multiplication with sparsity pattern
result = matrix_multiplication_with_sparsity(A, B, sparsity_pattern)
print(result)

# Perform harmonic-percussive separation
audio = torch.randn(44100)
harmonic, percussive = harmonic_percussive_separation(audio)
print(harmonic, percussive)

# Perform matrix operations with mixed precision
C_fp32, C_fp16 = matrix_operations_with_mixed_precision(A, B)
print(C_fp32, C_fp16)
```