
import torch

def torch_rms_energy_int8(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculates the root mean square (RMS) energy of an input tensor, quantized to int8.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    
    # Square the input tensor
    squared_input = torch.square(input_int8)

    # Multiply with weight tensor
    weighted_input = squared_input * weight_int8

    # Sum across all dimensions
    summed_input = torch.sum(weighted_input)

    # Calculate the RMS energy
    rms_energy = torch.sqrt(summed_input / torch.numel(input_tensor))

    return rms_energy.to(torch.float32)

function_signature = {
    "name": "torch_rms_energy_int8",
    "inputs": [
        ((10, 10), torch.float32),
        ((10, 10), torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
