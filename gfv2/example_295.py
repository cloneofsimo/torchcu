
import torch

def adaptive_avg_pool_permute_bilinear_fp16_int8(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs adaptive average pooling, permutation, bilinear interpolation, and quantization.
    """
    # Adaptive average pooling
    input_tensor = torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))

    # Permutation
    input_tensor = input_tensor.permute(0, 2, 3, 1)

    # Bilinear interpolation
    output = torch.nn.functional.bilinear(input_tensor.float(), weight.float(), output_size=(8, 8))

    # Convert to fp16
    output = output.to(torch.float16)

    # Quantize to int8
    output = torch.quantize_per_tensor(output, scale=1.0, zero_point=0, dtype=torch.quint8)
    return output

function_signature = {
    "name": "adaptive_avg_pool_permute_bilinear_fp16_int8",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        ((3, 32, 32, 3), torch.float32)
    ],
    "outputs": [
        ((1, 8, 8, 3), torch.quint8),
    ]
}
