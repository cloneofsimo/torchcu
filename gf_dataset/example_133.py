
import torch

def torch_complex_filter_function(input_tensor: torch.Tensor, filter: torch.Tensor) -> torch.Tensor:
    """
    Applies a complex filter to a complex-valued input tensor using an inverse DFT (IDFT).
    """
    # Convert to complex tensors
    input_complex = torch.view_as_complex(input_tensor)
    filter_complex = torch.view_as_complex(filter)

    # Perform IDFT
    filtered_complex = torch.fft.ifft(input_complex * filter_complex)

    # Return the real component of the filtered output
    return torch.view_as_real(filtered_complex)[:, :, 0]

function_signature = {
    "name": "torch_complex_filter_function",
    "inputs": [
        ((2, 3, 4, 2), torch.float32),
        ((4, 2), torch.float32)
    ],
    "outputs": [
        ((2, 3, 4), torch.float32),
    ]
}
