
import torch
import torch.nn.functional as F

@torch.no_grad()
def torch_softplus_time_stretch_grad_checkpointing(input_tensor: torch.Tensor, time_stretch_factor: float) -> torch.Tensor:
    """
    Applies the softplus activation function to the input tensor, then performs time stretching, 
    and finally uses gradient checkpointing for efficiency.
    """

    def softplus_stretch_fn(input_tensor):
        output = F.softplus(input_tensor, beta=1.0)  # Inplace activation
        output = torch.nn.functional.interpolate(output, scale_factor=time_stretch_factor, mode='linear')
        return output

    # Gradient checkpointing for the entire function
    output = torch.utils.checkpoint.checkpoint(softplus_stretch_fn, input_tensor) 
    return output

function_signature = {
    "name": "torch_softplus_time_stretch_grad_checkpointing",
    "inputs": [
        ((2, 3, 4, 5), torch.float32),
        (torch.float32),
    ],
    "outputs": [
        ((2, 3, int(4*time_stretch_factor), 5), torch.float32),
    ]
}
