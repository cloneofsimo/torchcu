
import torch
import torch.nn.functional as F

def complex_image_processing(image_tensor: torch.Tensor, filter_kernel: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Processes an image tensor using a series of operations:
    1. Applies a median filter to the image.
    2. Performs element-wise summation with a filter kernel.
    3. Applies a greater than or equal comparison with a threshold (0.5).
    4. Applies softmax with a temperature.
    5. Converts the output to int8.
    """
    
    # Median filter
    filtered_image = torch.median(image_tensor, dim=1, keepdim=True).values
    
    # Element-wise sum
    summed_image = filtered_image + filter_kernel
    
    # Greater than or equal comparison
    thresholded_image = (summed_image >= 0.5).to(torch.float32)
    
    # Softmax with temperature
    softmaxed_image = F.softmax(thresholded_image * temperature, dim=1)
    
    # Convert to int8
    int8_image = softmaxed_image.to(torch.int8)
    
    return int8_image

function_signature = {
    "name": "complex_image_processing",
    "inputs": [
        ((3, 32, 32), torch.float32),
        ((1, 32, 32), torch.float32),
    ],
    "outputs": [
        ((3, 32, 32), torch.int8),
    ]
}
