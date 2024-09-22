
import torch
import torch.nn.functional as F

def torch_dwt_bucketize_sum_int8(input_tensor: torch.Tensor, wavelet: str, levels: int, buckets: int) -> torch.Tensor:
    """
    Performs inverse discrete wavelet transform (IDWT), bucketizes the result, sums within each bucket, and returns the sum in int8.
    """
    # IDWT
    output = torch.idwt(input_tensor, wavelet, levels)
    # Bucketize
    output = torch.bucketize(output, torch.linspace(0, 1, buckets + 1))
    # Sum within buckets
    output = torch.bincount(output, minlength=buckets)
    # Convert to int8
    output = output.to(torch.int8)
    return output

function_signature = {
    "name": "torch_dwt_bucketize_sum_int8",
    "inputs": [
        ((16, 16), torch.float32),
        ("db1", str),
        (3, int),
        (10, int)
    ],
    "outputs": [
        ((10,), torch.int8),
    ]
}
