
import torch

def torch_hadamard_product_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs hadamard product between input_tensor and weight, then repeats the result along
    the first dimension, takes the kth value along the second dimension, and finally returns
    the result converted to fp16.
    """
    # Perform hadamard product
    hadamard_product = input_tensor.to(torch.float16) * weight.to(torch.float16)

    # Repeat the product along the first dimension
    repeated_product = hadamard_product.repeat(2, 1)

    # Find the kth value along the second dimension (k = 3 in this example)
    k = 3
    kth_values, _ = torch.kthvalue(repeated_product, k, dim=1)

    # Return the kth values as fp16
    return kth_values.to(torch.float16)

function_signature = {
    "name": "torch_hadamard_product_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((8, 4), torch.float16),
    ]
}
