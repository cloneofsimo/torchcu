import torch
import torch.backends.cudnn

def cudnn_fp16_support(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Checks if CUDNN and FP16 support are available and returns the input tensor in FP16 format.

    Args:
        input_tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor in FP16 format.
    """
    # Check if CUDNN and FP16 support are available
    if torch.backends.cudnn.is_available() and torch.cuda.is_available():
        # Move the tensor to the GPU
        input_tensor = input_tensor.to("cuda")

        # Convert the tensor to FP16 format
        input_tensor = input_tensor.half()

    return input_tensor



# function_signature
function_signature = {
    "name": "cudnn_fp16_support",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}