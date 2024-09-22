
import torch

def permute_round_grad_accumulate_bf16(input_tensor: torch.Tensor, weight: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Performs the following operations:
    1. Permutes the input tensor.
    2. Rounds the permuted tensor to nearest integer.
    3. Performs a matrix multiplication with the weight tensor in bfloat16.
    4. Accumulates the gradient with the scaled weight tensor.
    5. Returns the result in fp32.
    """
    input_tensor_permuted = input_tensor.permute(1, 0, 2)
    input_tensor_rounded = torch.round(input_tensor_permuted).to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_tensor_rounded, weight_bf16.t())
    output.backward(torch.ones_like(output))
    weight.grad.data += scale * weight.grad.data
    return output.to(torch.float32)

function_signature = {
    "name": "permute_round_grad_accumulate_bf16",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((4, 5), torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((2, 5), torch.float32),
    ]
}
