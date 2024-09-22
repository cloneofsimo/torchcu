
import torch

def cosine_similarity_with_padding(input_tensor: torch.Tensor, query_tensor: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    """
    Calculate cosine similarity between input tensor and query tensor after padding.
    Input tensor and query tensor are assumed to have the same number of dimensions.
    """
    # Ensure both tensors have the same number of dimensions
    assert input_tensor.ndim == query_tensor.ndim, "Input and query tensors must have the same number of dimensions"

    # Calculate the maximum size of each dimension
    max_sizes = [max(dim_size1, dim_size2) for dim_size1, dim_size2 in zip(input_tensor.shape, query_tensor.shape)]

    # Pad input tensor
    input_tensor = torch.nn.functional.pad(input_tensor, [0, max_sizes[1] - input_tensor.shape[1], 0, max_sizes[0] - input_tensor.shape[0]], "constant", value=padding_value)

    # Pad query tensor
    query_tensor = torch.nn.functional.pad(query_tensor, [0, max_sizes[1] - query_tensor.shape[1], 0, max_sizes[0] - query_tensor.shape[0]], "constant", value=padding_value)

    # Squeeze both tensors to remove singleton dimensions
    input_tensor = input_tensor.squeeze()
    query_tensor = query_tensor.squeeze()

    # Calculate cosine similarity
    output = torch.nn.functional.cosine_similarity(input_tensor.float(), query_tensor.float(), dim=0)
    return output

function_signature = {
    "name": "cosine_similarity_with_padding",
    "inputs": [
        ((2, 3), torch.int8),
        ((2, 3), torch.int8),
        ((), torch.int32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
