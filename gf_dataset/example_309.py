
import torch

def torch_glu_bilinear_fp16_function(input_tensor: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a Gated Linear Unit (GLU) followed by a bilinear transformation in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight1_fp16 = weight1.to(torch.float16)
    weight2_fp16 = weight2.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)

    # GLU
    linear1 = torch.nn.functional.linear(input_fp16, weight1_fp16, bias_fp16)
    linear2 = torch.nn.functional.linear(input_fp16, weight2_fp16, bias_fp16)
    glu_output = linear1 * torch.sigmoid(linear2)

    # Bilinear
    output = torch.bmm(glu_output.unsqueeze(1), weight2_fp16.unsqueeze(0)).squeeze(1)

    return output.to(torch.float32)

function_signature = {
    "name": "torch_glu_bilinear_fp16_function",
    "inputs": [
        ((4, 8), torch.float32),
        ((8, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32)
    ]
}
