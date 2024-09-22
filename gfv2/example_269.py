
import torch
import torch.nn.functional as F

def transposed_conv3d_rank_int8(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Performs a transposed 3D convolution, calculates the matrix rank of the output, and returns the result as int8.
    """
    output = F.conv_transpose3d(input_tensor.to(torch.float32), weight.to(torch.float32), bias=bias.to(torch.float32))
    output.to(torch.int8, inplace=True)  # Inplace conversion to int8
    rank = torch.matrix_rank(output)  # Calculate matrix rank
    return output, int(rank)  # Return output and rank as int

function_signature = {
    "name": "transposed_conv3d_rank_int8",
    "inputs": [
        ((1, 2, 3, 4, 5), torch.float32),
        ((2, 2, 3, 3, 3), torch.float32),
        ((2,), torch.float32),
    ],
    "outputs": [
        ((1, 2, 4, 6, 7), torch.int8),
        ((), torch.int32),
    ]
}
