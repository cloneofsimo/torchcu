
import torch
import numpy as np

def torch_load_image_normalize_function(image_path: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Loads an image from a file, normalizes it using provided mean and std, and returns the normalized image.
    """
    image = torch.from_numpy(np.load(image_path))
    image = (image - mean) / std
    return image

function_signature = {
    "name": "torch_load_image_normalize_function",
    "inputs": [
        ((" ", ), str),
        ((3, ), torch.float32),
        ((3, ), torch.float32)
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),
    ]
}
