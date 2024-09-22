
import torch
import torch.nn.functional as F

def pixel_shuffle_mixup_int8_fp16(input_tensor: torch.Tensor, gt: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Performs pixel shuffle, mixup with given alpha, and converts to int8.
    Applies a ReLU activation on the result and returns it in fp16 format.
    """
    # Pixel Shuffle
    input_tensor = F.pixel_shuffle(input_tensor, upscale_factor=2)

    # Mixup
    lam = torch.rand(1).to(input_tensor.device)
    input_tensor = lam * input_tensor + (1 - lam) * gt

    # Convert to int8 and apply ReLU
    input_tensor = input_tensor.to(torch.int8)
    input_tensor = F.relu(input_tensor)

    # Convert to fp16
    input_tensor = input_tensor.to(torch.float16)

    return input_tensor

function_signature = {
    "name": "pixel_shuffle_mixup_int8_fp16",
    "inputs": [
        ((16, 3, 32, 32), torch.float32),
        ((16, 3, 32, 32), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((16, 3, 64, 64), torch.float16),
    ]
}
