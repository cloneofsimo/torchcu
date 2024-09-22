
import torch

def torch_diagflat_einsum_backward(input_tensor: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a series of operations:
    1. Creates a diagonal matrix from the input tensor.
    2. Transposes the weight tensor.
    3. Performs matrix multiplication between the diagonal matrix and transposed weight.
    4. Calculates the backward pass for the matrix multiplication.
    """

    diag_matrix = torch.diagflat(input_tensor)
    weight_t = weight.t()
    output = torch.einsum("ij,jk->ik", diag_matrix, weight_t)

    # Backward pass (calculate gradients)
    grad_output = torch.ones_like(output)
    grad_diag, grad_weight_t = torch.autograd.grad(
        outputs=output,
        inputs=[diag_matrix, weight_t],
        grad_outputs=grad_output,
        create_graph=True,
        retain_graph=True
    )
    grad_weight = grad_weight_t.t()

    return output, (grad_diag, grad_weight)

function_signature = {
    "name": "torch_diagflat_einsum_backward",
    "inputs": [
        ((4,), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
        [((4,), torch.float32), ((4, 4), torch.float32)]
    ]
}
