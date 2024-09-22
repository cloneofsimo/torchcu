
import torch

def complex_transform_function(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex transformation on the input tensor using a specified kernel.
    This involves:
    1. Reflecting the input tensor along the edges.
    2. Convolving the reflected input with the kernel using einsum_inner for efficiency.
    3. Converting the result to fp16 for reduced memory usage.
    4. Applying a custom activation function.
    5. Converting the result back to bf16 before returning.
    """
    # 1. Reflection padding
    input_padded = torch.nn.functional.pad(input_tensor, (kernel.shape[2] // 2, kernel.shape[2] // 2,
                                                        kernel.shape[1] // 2, kernel.shape[1] // 2),
                                         mode='reflect')

    # 2. Convolution using einsum_inner
    output = torch.einsum("bhw,khw->b(hw)", input_padded, kernel)

    # 3. Convert to fp16
    output = output.to(torch.float16)

    # 4. Custom activation function (example)
    output = torch.sigmoid(output)

    # 5. Convert to bf16 and return
    return output.to(torch.bfloat16)

function_signature = {
    "name": "complex_transform_function",
    "inputs": [
        ((1, 3, 10, 10), torch.float32),  # Input tensor
        ((3, 3, 3, 3), torch.float32)  # Kernel
    ],
    "outputs": [
        ((1, 3, 10, 10), torch.bfloat16),  # Output tensor
    ]
}
