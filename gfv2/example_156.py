
import torch
import torch.fft

def complex_bilinear_interp_bf16(input_tensor: torch.Tensor, 
                                  weights: torch.Tensor, 
                                  grid: torch.Tensor, 
                                  output_size: list[int], 
                                  mode: str = "bilinear") -> torch.Tensor:
    """
    Performs complex bilinear interpolation on a batch of 2D complex-valued tensors.
    
    Args:
        input_tensor: A 4D tensor of shape (batch_size, 2, height, width), where the second dimension
                      represents the real and imaginary parts of the complex values.
        weights: A 3D tensor of shape (batch_size, height, width), representing the weights for each
                 input location.
        grid: A 4D tensor of shape (batch_size, 2, height, width), representing the coordinates of the 
              output grid points in the input space. The second dimension represents (x, y) coordinates.
        output_size: A list of two integers representing the desired height and width of the output.
        mode: The interpolation mode, can be "bilinear" (default) or "nearest".

    Returns:
        A 4D tensor of shape (batch_size, 2, output_height, output_width) representing the interpolated
        complex values.
    """

    # Convert to bfloat16 for faster computations
    input_bf16 = input_tensor.to(torch.bfloat16)
    weights_bf16 = weights.to(torch.bfloat16)
    grid_bf16 = grid.to(torch.bfloat16)

    # Perform complex bilinear interpolation
    output_bf16 = torch.nn.functional.grid_sample(input_bf16, grid_bf16, mode=mode, align_corners=False)

    # Multiply by weights
    output_bf16 = output_bf16 * weights_bf16.unsqueeze(1)

    # Convert back to float16
    output_fp16 = output_bf16.to(torch.float16)

    # Reshape and perform inverse FFT shift
    output_fp16 = output_fp16.reshape(input_tensor.shape[0], 2, *output_size)
    output_fp16 = torch.fft.ifftshift(output_fp16, dim=[2, 3])

    return output_fp16

function_signature = {
    "name": "complex_bilinear_interp_bf16",
    "inputs": [
        ((1, 2, 128, 128), torch.complex64),
        ((1, 128, 128), torch.float32),
        ((1, 2, 128, 128), torch.float32),
        ( (128, 128), torch.int32),
        ("bilinear", torch.int32),
    ],
    "outputs": [
        ((1, 2, 128, 128), torch.float16),
    ]
}

