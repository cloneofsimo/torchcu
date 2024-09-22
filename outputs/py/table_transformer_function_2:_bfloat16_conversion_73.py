import torch

def bfloat16_conversion(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts the input tensor to BFloat16 format.

    Args:
        input_tensor (torch.Tensor): The input tensor to convert to BFloat16.

    Returns:
        torch.Tensor: The input tensor converted to BFloat16.
    """
    # Convert the input tensor to BFloat16
    bfloat16_out = input_tensor.to(torch.bfloat16)

    return bfloat16_out



# function_signature
function_signature = {
    "name": "bfloat16_conversion",
    "inputs": [
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.bfloat16)]
}