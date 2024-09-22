
import torch

def my_function(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on the input tensor.
    """
    # Convert to fp16
    input_fp16 = input_tensor.to(torch.float16)

    # Calculate non-zero indices
    non_zero_indices = torch.nonzero(input_fp16)

    # Extract values at non-zero indices
    non_zero_values = input_fp16[non_zero_indices]

    # Clamp values to a range
    clamped_values = torch.clamp(non_zero_values, min=-1.0, max=1.0)

    # Create a new tensor based on clamped values
    output_tensor = torch.zeros_like(input_tensor)
    output_tensor[non_zero_indices] = clamped_values

    # Convert back to fp32
    output_tensor = output_tensor.to(torch.float32)

    # Perform a comparison and return the result
    return output_tensor.ne(0).float()

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
