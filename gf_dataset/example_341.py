
import torch

def torch_kth_value_linear_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs a linear transformation on the input tensor, then returns the k-th smallest value along the first dimension.
    The computation is done using FP16 precision for efficiency.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    output = torch.matmul(input_fp16, weight_fp16.t())
    kth_value = torch.kthvalue(output, k, dim=0)[0]  # Returns only the k-th value, not the index
    return kth_value.to(torch.float32)  # Return the result in FP32

function_signature = {
    "name": "torch_kth_value_linear_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (1, torch.int32)
    ],
    "outputs": [
        ((4,), torch.float32),
    ]
}
