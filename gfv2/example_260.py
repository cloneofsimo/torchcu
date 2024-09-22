
import torch

def complex_bfloat16_function(input1: torch.Tensor, input2: torch.Tensor, input3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a complex operation involving bfloat16, matrix multiplication, and element-wise operations.

    Args:
        input1: A tensor of shape (N, C, H, W).
        input2: A tensor of shape (M, K).
        input3: A tensor of shape (N, C, H, W).

    Returns:
        A tuple containing two tensors:
            - output1: A tensor of shape (N, M, H, W).
            - output2: A tensor of shape (N, C, H, W).
    """

    input1_bf16 = input1.to(torch.bfloat16)
    input2_bf16 = input2.to(torch.bfloat16)
    input3_bf16 = input3.to(torch.bfloat16)

    # Matrix multiplication with bfloat16
    output1_bf16 = torch.bmm(input1_bf16.view(-1, input1.size(1), input1.size(2) * input1.size(3)), input2_bf16.t()).view(input1.size(0), input2.size(0), input1.size(2), input1.size(3))

    # Element-wise addition with bfloat16
    output2_bf16 = torch.baddbmm(input3_bf16, input1_bf16, input2_bf16, beta=1.0, alpha=1.0)

    # Cross-entropy calculation with bfloat16
    output3_bf16 = torch.nn.functional.cross_entropy(output1_bf16, input3_bf16)

    # Convert results back to float32
    output1 = output1_bf16.to(torch.float32)
    output2 = output2_bf16.to(torch.float32)

    return output1, output2

function_signature = {
    "name": "complex_bfloat16_function",
    "inputs": [
        ((1, 1, 1, 1), torch.float32),
        ((1, 1), torch.float32),
        ((1, 1, 1, 1), torch.float32)
    ],
    "outputs": [
        ((1, 1, 1, 1), torch.float32),
        ((1, 1, 1, 1), torch.float32)
    ]
}
