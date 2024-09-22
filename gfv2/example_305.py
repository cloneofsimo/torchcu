
import torch

def complex_tensor_operation(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of operations on a tensor, including:
    - Repeat the input tensor along a specific dimension
    - Perform an element-wise multiplication and addition (addcmul)
    - Quantize the result to int8 and then back to fp16
    - Apply a 1D convolution with specified kernel size and stride
    - Perform bucket-based quantization and cast to bfloat16
    - Finally, return the result.
    """

    # Repeat the input tensor along dimension 1
    repeated_tensor = input_tensor.repeat(1, 3, 1)

    # Element-wise multiplication and addition
    output = torch.addcmul(repeated_tensor, bias, weight)

    # Quantize to int8 and then back to fp16
    output_int8 = output.to(torch.int8)
    output_fp16 = output_int8.to(torch.float16)

    # 1D convolution
    kernel = torch.randn(1, 1, 3, dtype=torch.float16)
    output = torch.nn.functional.conv1d(output_fp16, kernel, stride=2)

    # Bucket-based quantization and cast to bfloat16
    buckets = torch.arange(-1.0, 1.0, 0.2, dtype=torch.float16)
    output = torch.bucketize(output, buckets).to(torch.bfloat16)

    return output

function_signature = {
    "name": "complex_tensor_operation",
    "inputs": [
        ((1, 1, 10), torch.float32),
        ((1, 1, 10), torch.float32),
        ((1, 1, 1), torch.float32),
    ],
    "outputs": [
        ((1, 1, 5), torch.bfloat16),
    ]
}
