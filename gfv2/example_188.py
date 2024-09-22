
import torch

def signal_processing_function(input_tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Processes a 1D signal by applying a rolling window, dot product with a filter,
    calculating the signal envelope, and applying an exponential function.
    """
    input_fp16 = input_tensor.to(torch.float16)
    rolled_tensor = torch.roll(input_fp16, -window_size, dims=0)  # Roll the signal
    filter_tensor = torch.ones(window_size, dtype=torch.float16) / window_size  # Simple averaging filter
    dot_product = torch.dot(rolled_tensor, filter_tensor)  # Calculate the dot product
    envelope = torch.abs(dot_product)  # Calculate the signal envelope
    output_tensor = torch.exp(-envelope)  # Apply exponential function
    return output_tensor.to(torch.float32)

function_signature = {
    "name": "signal_processing_function",
    "inputs": [
        ((100,), torch.float32),
        (int,)
    ],
    "outputs": [
        ((100,), torch.float32),
    ]
}
