
import torch
import torch.nn.functional as F

def pixel_shuffle_geglu_fp16(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a pixel shuffle, followed by a GEGLU activation, and a linear transformation.
    """
    # Convert input and weight to fp16
    input_tensor = input_tensor.to(torch.float16)
    weight = weight.to(torch.float16)

    # Pixel shuffle
    output = F.pixel_shuffle(input_tensor, upscale_factor=2)

    # GEGLU activation
    output = output * torch.sigmoid(output)

    # Linear transformation
    output = torch.matmul(output, weight.t())

    # Convert output back to fp32
    output = output.to(torch.float32)
    return output

function_signature = {
    "name": "pixel_shuffle_geglu_fp16",
    "inputs": [
        ((1, 16, 8, 8), torch.float32),  # Example input shape
        ((32, 32), torch.float32)       # Example weight shape
    ],
    "outputs": [
        ((1, 32, 16, 16), torch.float32),  # Example output shape
    ]
}
