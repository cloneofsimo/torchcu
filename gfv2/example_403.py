
import torch

def process_data(input_tensor: torch.Tensor, filename: str) -> torch.Tensor:
    """
    Loads data from a file, performs a simple operation, and returns the result.

    Args:
        input_tensor: A tensor of arbitrary size and type (at least 1 element).
        filename: Path to the file containing the data.

    Returns:
        A tensor of the same size as input_tensor, with each element multiplied by the sum of the loaded data.
    """
    data = torch.from_numpy(np.load(filename))  # Load data from file
    sum_data = data.sum().to(input_tensor.dtype)  # Calculate sum and convert to input dtype
    output = input_tensor * sum_data  # Multiply each element by the sum
    return output.to(torch.float16)  # Convert output to fp16

function_signature = {
    "name": "process_data",
    "inputs": [
        ((1,), torch.float32),  # Input tensor
        ("dummy", str),  # Filename (dummy type for now)
    ],
    "outputs": [
        ((1,), torch.float16),  # Output tensor
    ]
}
