
import torch
import torch.nn.functional as F

def quantized_bucketing_with_gradient_clipping(input_tensor: torch.Tensor, buckets: list,
                                                gradient_clip_value: float = 1.0) -> torch.Tensor:
    """
    Performs quantized bucketing with gradient clipping.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        buckets (list): List of bucket boundaries.
        gradient_clip_value (float): Maximum value for gradient clipping. Defaults to 1.0.

    Returns:
        torch.Tensor: Quantized and bucketized output tensor.
    """

    # Convert input to int8
    input_int8 = input_tensor.to(torch.int8)

    # Bucketize
    output_int8 = torch.bucketize(input_int8, torch.tensor(buckets, dtype=torch.int8))

    # Convert back to float32 for gradient clipping
    output_fp32 = output_int8.to(torch.float32)

    # Apply hardtanh with gradient clipping
    output_fp32 = F.hardtanh(output_fp32, min_val=-gradient_clip_value, max_val=gradient_clip_value)

    # Convert back to int8 for output
    output_int8 = output_fp32.to(torch.int8)

    return output_int8

function_signature = {
    "name": "quantized_bucketing_with_gradient_clipping",
    "inputs": [
        ((1,), torch.float32),
        ([], torch.int32)
    ],
    "outputs": [
        ((1,), torch.int8),
    ]
}
