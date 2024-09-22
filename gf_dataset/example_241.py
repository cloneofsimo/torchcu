
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

@custom_fwd(cast_inputs=torch.bfloat16)
def torch_image_topk_gradient_bf16(image: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the top-k values and their indices in an image tensor and computes gradients for the top-k values.
    """
    # Cast to bfloat16 for efficiency
    image_bf16 = image.to(torch.bfloat16)
    
    # Find top-k values and indices
    topk_values, topk_indices = torch.topk(image_bf16, k=k, dim=1)
    
    # Compute image gradient
    image_gradient = torch.zeros_like(image, dtype=torch.float32)
    image_gradient.scatter_(1, topk_indices, topk_values.to(torch.float32))
    
    return image_gradient, topk_indices

@custom_bwd
def bwd_torch_image_topk_gradient_bf16(ctx, grad_output, grad_topk_indices):
    """
    Backwards pass for the function, applying gradients to the top-k values.
    """
    # Retrieve the inputs
    image_bf16 = ctx.inputs[0]
    k = ctx.inputs[1]
    
    # Gradient of top-k values is simply the gradient of the output
    grad_topk_values = grad_output
    
    # Compute the gradient of the input image
    grad_input = torch.zeros_like(image_bf16, dtype=torch.bfloat16)
    grad_input.scatter_(1, ctx.outputs[1], grad_topk_values)
    
    # Return the gradients
    return grad_input, None

function_signature = {
    "name": "torch_image_topk_gradient_bf16",
    "inputs": [
        ((3, 224, 224), torch.float32),  # Image tensor (batch, height, width)
        ((), torch.int32)  # Number of top-k values
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),  # Gradient tensor (batch, height, width)
        ((3, 10), torch.int64),  # Indices of top-k values
    ]
}
