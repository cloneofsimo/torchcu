
import torch

def torch_bce_with_logits_bmm_fp16(input_tensor: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a batch matrix multiplication (bmm) followed by binary cross-entropy with logits, all in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    target_fp16 = target.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)

    output = torch.bmm(input_fp16, weight_fp16.t())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target_fp16)
    return loss

function_signature = {
    "name": "torch_bce_with_logits_bmm_fp16",
    "inputs": [
        ((2, 3, 4), torch.float32),
        ((2, 4, 5), torch.float32),
        ((3, 5), torch.float32)
    ],
    "outputs": [
        ((2, 3, 5), torch.float32),
    ]
}
