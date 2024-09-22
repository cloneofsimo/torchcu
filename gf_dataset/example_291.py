
import torch
import torch.nn.functional as F

def torch_conv_similarity_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Performs a convolution, applies tanh activation, calculates cosine similarity with a weight tensor, 
    and returns the similarity scores. 
    """
    # Convolution using torch.nn.functional.conv_tbc
    output = F.conv_tbc(input_tensor.unsqueeze(1), weight.unsqueeze(1))
    # Apply tanh activation
    output = torch.tanh(output)
    # Calculate cosine similarity
    similarity = F.cosine_similarity(output, weight.unsqueeze(1), dim=1)
    # Return similarity scores
    return similarity

function_signature = {
    "name": "torch_conv_similarity_function",
    "inputs": [
        ((10, 5, 10), torch.float32),
        ((5, 1, 3), torch.float32)
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
