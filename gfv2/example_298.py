
import torch

def pixel_shuffle_einsum_transpose(input_tensor: torch.Tensor, upscale_factor: int) -> list[torch.Tensor]:
    """
    Performs pixel shuffle, einsum transpose, and returns both the shuffled tensor and transposed tensor.
    """
    # Pixel Shuffle
    shuffled_tensor = torch.pixel_shuffle(input_tensor, upscale_factor)

    # Einsum Transpose
    transposed_tensor = torch.einsum('bhwc->bhcw', shuffled_tensor)

    return [shuffled_tensor, transposed_tensor]


function_signature = {
    "name": "pixel_shuffle_einsum_transpose",
    "inputs": [
        ((1, 16, 4, 4), torch.float32),  # Input tensor
        (int, None)  # Upscale factor
    ],
    "outputs": [
        ((1, 4, 16, 16), torch.float32),  # Shuffled tensor
        ((1, 16, 16, 4), torch.float32)  # Transposed tensor
    ]
}
