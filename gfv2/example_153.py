
import torch

def einsum_addr_bf16_function(input1: torch.Tensor, input2: torch.Tensor, input3: torch.Tensor) -> torch.Tensor:
    """
    Performs a batched einsum with inner product followed by addition with a third tensor. 
    All operations are performed in bfloat16 for potential performance gains. 
    """
    input1_bf16 = input1.to(torch.bfloat16)
    input2_bf16 = input2.to(torch.bfloat16)
    input3_bf16 = input3.to(torch.bfloat16)

    output_bf16 = torch.einsum('bij,bjk->bik', input1_bf16, input2_bf16) 
    output_bf16 = output_bf16 + input3_bf16
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "einsum_addr_bf16_function",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 4, 5), torch.float32),
        ((2, 3, 5), torch.float32)
    ],
    "outputs": [
        ((2, 3, 5), torch.float32),
    ]
}
