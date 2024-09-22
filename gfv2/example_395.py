
import torch

def my_function(input_tensor: torch.Tensor, buckets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs the following operations on input tensor:
        1. Bucketizes input tensor based on provided buckets
        2. Inverts the bucketized tensor
        3. Applies tanh activation to inverted tensor
        4. Converts the result to FP16 
        5. Returns the activated tensor and the original bucketized tensor 
    """
    # Bucketize
    bucketized_tensor = torch.bucketize(input_tensor, buckets)
    # Invert
    inverted_tensor = 1 - bucketized_tensor
    # Tanh activation
    activated_tensor = torch.tanh(inverted_tensor.to(torch.float32))
    # FP16 conversion
    activated_tensor = activated_tensor.to(torch.float16)
    
    return activated_tensor, bucketized_tensor

function_signature = {
    "name": "my_function",
    "inputs": [
        ((1,), torch.float32), # Input tensor should have at least 1 element
        ((10,), torch.float32), # Buckets tensor
    ],
    "outputs": [
        ((1,), torch.float16),
        ((1,), torch.int64),
    ]
}
