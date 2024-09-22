
import torch
import torch.nn.functional as F

def torch_sharpen_with_fading_out(input_tensor: torch.Tensor, sharpness_factor: float, fading_out_factor: float) -> torch.Tensor:
    """
    Sharpens an image by applying a Scharr gradient filter and then fades out the edges using a sigmoid function.
    
    Args:
        input_tensor: The input image tensor.
        sharpness_factor:  A factor to control the sharpness of the Scharr gradient.
        fading_out_factor: A factor to control the fading out of the edges.

    Returns:
        The sharpened and faded image tensor.
    """
    # Calculate Scharr gradient
    gradient_x = F.scharr(input_tensor, (1, 0))
    gradient_y = F.scharr(input_tensor, (0, 1))

    # Calculate the magnitude of the gradient
    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Apply sharpness factor
    sharp_gradient = gradient_magnitude * sharpness_factor

    # Calculate sigmoid fading function
    fading_out = torch.sigmoid(sharp_gradient * fading_out_factor)

    # Apply fading out to the original image
    output = input_tensor * fading_out

    return output

function_signature = {
    "name": "torch_sharpen_with_fading_out",
    "inputs": [
        ((3, 224, 224), torch.float32),
        (torch.float32),
        (torch.float32)
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),
    ]
}
