
import torch
import torch.nn.functional as F

def torch_l1_loss_bilinear_fftshift(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a sequence of operations:
    1. Calculates the L1 loss between two input tensors.
    2. Applies a bilinear operation with the given weight tensor.
    3. Performs a FFT shift.
    """
    l1_loss = F.l1_loss(input_tensor1, input_tensor2, reduction='none')  # Calculate L1 loss
    bilinear_output = F.bilinear(l1_loss.float(), input_tensor1.float(), weight.float())  # Bilinear operation
    fftshift_output = torch.fft.fftshift(bilinear_output, dim=(-2, -1))  # FFT shift
    return fftshift_output

function_signature = {
    "name": "torch_l1_loss_bilinear_fftshift",
    "inputs": [
        ((8, 8), torch.int8),
        ((8, 8), torch.int8),
        ((8, 8, 8, 8), torch.float32)
    ],
    "outputs": [
        ((8, 8, 8, 8), torch.float32)
    ]
}
