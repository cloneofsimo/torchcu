
import torch
import torch.nn.functional as F

def torch_glu_fused_layernorm_with_gradient_magnitude(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                                                     gamma: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a Gated Linear Unit (GLU) activation with fused Layer Normalization and gradient magnitude calculation.

    Args:
        input_tensor: Input tensor of shape (batch_size, seq_len, hidden_size).
        weight: Weight tensor for the linear transformation of shape (hidden_size, hidden_size).
        bias: Bias tensor for the linear transformation of shape (hidden_size).
        gamma: Scaling factor for Layer Normalization of shape (hidden_size).
        beta: Shifting factor for Layer Normalization of shape (hidden_size).

    Returns:
        A tuple containing:
            - The output tensor of shape (batch_size, seq_len, hidden_size).
            - The gradient magnitude of the input tensor of shape (batch_size, seq_len, hidden_size).
    """

    # Linear transformation
    output = F.linear(input_tensor, weight, bias)

    # Fused Layer Normalization
    output = F.layer_norm(output, output.shape[-1], eps=1e-5, elementwise_affine=True,
                           weight=gamma, bias=beta)

    # Gated Linear Unit
    output = output * torch.sigmoid(output)

    # Gradient magnitude calculation
    grad_magnitude = torch.abs(torch.autograd.grad(output, input_tensor, grad_outputs=torch.ones_like(output),
                                                 create_graph=True, retain_graph=True)[0])

    return output, grad_magnitude

function_signature = {
    "name": "torch_glu_fused_layernorm_with_gradient_magnitude",
    "inputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32)
    ],
    "outputs": [
        ((4, 4, 4), torch.float32),
        ((4, 4, 4), torch.float32)
    ]
}
