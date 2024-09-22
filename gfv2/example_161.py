
import torch
import torch.nn.functional as F

def coord_attention_transposed_conv2d_int8(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, output_padding: int) -> torch.Tensor:
    """
    Performs a transposed convolution, followed by coordinate attention and a fused softmax operation, all in int8 precision.
    """
    # 1. Transposed Convolution (int8)
    input_int8 = input_tensor.to(torch.int8)
    weight_int8 = weight.to(torch.int8)
    bias_int8 = bias.to(torch.int8)
    output_int8 = F.conv_transpose2d(input_int8, weight_int8, bias_int8, stride=stride, padding=padding, output_padding=output_padding)

    # 2. Coordinate Attention (int8)
    batch_size, channels, height, width = output_int8.shape
    x = output_int8.float()  # Convert to FP32 for coordinate attention calculations
    x_h = x.mean(dim=2, keepdim=True)
    x_w = x.mean(dim=3, keepdim=True)

    h_att = torch.matmul(x_h.permute(0, 2, 1), x_w).float()  # Matrix multiplication for attention calculation
    w_att = torch.matmul(x_w.permute(0, 3, 1), x_h).float()

    h_att = F.softmax(h_att, dim=2)
    w_att = F.softmax(w_att, dim=3)

    x_h_att = torch.matmul(h_att, x_h.permute(0, 1, 2)).permute(0, 2, 1)
    x_w_att = torch.matmul(w_att, x_w.permute(0, 1, 3)).permute(0, 2, 1)

    output_int8 = (x_h_att + x_w_att).to(torch.int8)

    # 3. Fused Softmax (int8)
    output_int8 = F.softmax(output_int8, dim=1, dtype=torch.int8)

    return output_int8.float()  # Return as FP32 for easier downstream processing

function_signature = {
    "name": "coord_attention_transposed_conv2d_int8",
    "inputs": [
        ((1, 32, 8, 8), torch.float32),
        ((32, 32, 3, 3), torch.float32),
        ((32,), torch.float32),
        (2, ),
        (1, ),
        (1, )
    ],
    "outputs": [
        ((1, 32, 16, 16), torch.float32),
    ]
}
