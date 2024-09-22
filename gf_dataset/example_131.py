
import torch

def torch_median_addcmul_clamp_int8(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on tensors, including median calculation, addcmul, clamp, and int8 conversion.
    """
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)

    # Calculate the median along dimension 1
    median_values = torch.median(input_int8, dim=1).values

    # Perform addcmul
    output_int8 = torch.addcmul(bias_int8, weight_int8, median_values, value=1)

    # Clamp the output
    output_int8 = torch.clamp(output_int8, min=-128, max=127)

    # Convert back to float32
    output = output_int8.to(torch.float32)

    return output

function_signature = {
    "name": "torch_median_addcmul_clamp_int8",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
