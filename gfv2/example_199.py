
import torch
import torch.nn.functional as F

def image_jacobian_function(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Computes the image Jacobian of a linear layer, optionally applying Swiglu activation.
    """
    
    # Cast to bfloat16
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)
    
    # Linear transformation
    output = torch.matmul(input_bf16, weight_bf16.t()) + bias_bf16

    # Swiglu activation
    output = F.swiglu(output)

    # Compute image Jacobian
    jacobian = torch.autograd.functional.jacobian(lambda x: F.swiglu(torch.matmul(x, weight_bf16.t()) + bias_bf16), input_bf16)
    
    # Flatten the Jacobian and convert to float32
    jacobian = jacobian.flatten(start_dim=1).to(torch.float32)

    # Return the Jacobian
    return jacobian

function_signature = {
    "name": "image_jacobian_function",
    "inputs": [
        ((4, 3, 224, 224), torch.float32), 
        ((1024, 3 * 224 * 224), torch.float32),
        ((1024,), torch.float32)
    ],
    "outputs": [
        ((4, 1024 * 3 * 224 * 224), torch.float32)
    ]
}

