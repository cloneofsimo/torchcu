
import torch
import torch.nn.functional as F

def multi_scale_attention_fp16(input_tensor: torch.Tensor,
                                  attention_weights: torch.Tensor,
                                  scales: list) -> torch.Tensor:
    """
    Performs multi-scale attention on the input tensor with provided attention weights.
    
    Args:
        input_tensor: The input tensor of shape (batch_size, channels, height, width).
        attention_weights: The attention weights tensor of shape (batch_size, num_scales, height, width).
        scales: A list of scaling factors for the multi-scale attention.

    Returns:
        The output tensor of shape (batch_size, channels, height, width) after multi-scale attention.
    """
    output = torch.zeros_like(input_tensor, dtype=torch.float16)
    for scale in scales:
        # Apply Laplace filter for the current scale
        filtered_input = F.conv2d(input_tensor.float(), torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).reshape(1, 1, 3, 3), padding=1).to(torch.float16)

        # Resize the attention weights to match the filtered input size
        resized_weights = F.interpolate(attention_weights, scale_factor=scale, mode='bilinear', align_corners=False)

        # Apply attention weights
        output += filtered_input * resized_weights

    return output.float()

function_signature = {
    "name": "multi_scale_attention_fp16",
    "inputs": [
        ((1, 3, 128, 128), torch.float32),
        ((1, 3, 128, 128), torch.float32),
        [2, 4, 8]
    ],
    "outputs": [
        ((1, 3, 128, 128), torch.float32),
    ]
}
