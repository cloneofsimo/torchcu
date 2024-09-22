
import torch

def torch_frobenius_norm_einsum_tanh_fp16(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Frobenius norm of the element-wise product of three tensors,
    then performs an einsum operation, followed by a tanh activation. All operations are done in fp16.
    """
    input_tensor1_fp16 = input_tensor1.to(torch.float16)
    input_tensor2_fp16 = input_tensor2.to(torch.float16)
    input_tensor3_fp16 = input_tensor3.to(torch.float16)

    element_wise_product = input_tensor1_fp16 * input_tensor2_fp16 * input_tensor3_fp16
    frobenius_norm = torch.linalg.norm(element_wise_product)

    einsum_result = torch.einsum('ij,jk->ik', input_tensor1_fp16, input_tensor2_fp16)
    output = torch.tanh(einsum_result)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_frobenius_norm_einsum_tanh_fp16",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 3, 4), torch.float32),
        ((2, 3, 4), torch.float32)
    ],
    "outputs": [
        ((2, 3), torch.float32),
    ]
}

