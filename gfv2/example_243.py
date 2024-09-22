
import torch

def complex_activation_function(input_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Applies a complex activation function to an input tensor, returning multiple outputs.

    This function combines various operations including:
    - Gelu activation
    - Determinant calculation
    - Matrix exponential
    - Mish activation
    - Type casting to FP16

    Returns a list of tensors:
    - Gelu(input_tensor).to(torch.float16)
    - det(input_tensor)
    - matrix_exp(input_tensor)
    - mish(input_tensor)
    """
    input_fp16 = input_tensor.to(torch.float16)
    gelu_output = torch.nn.functional.gelu(input_fp16)
    det_output = torch.linalg.det(input_fp16)
    matrix_exp_output = torch.matrix_exp(input_fp16)
    mish_output = torch.nn.functional.mish(input_fp16)
    return [gelu_output, det_output, matrix_exp_output, mish_output]

function_signature = {
    "name": "complex_activation_function",
    "inputs": [
        ((2, 2), torch.float32),
    ],
    "outputs": [
        ((2, 2), torch.float16),
        ((1,), torch.float32),
        ((2, 2), torch.float32),
        ((2, 2), torch.float16),
    ]
}
