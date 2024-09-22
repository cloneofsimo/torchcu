
import torch

def my_function(input_tensor: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """
    Sorts the input tensor along the first dimension based on the specified mode and returns
    the top k elements.
    """
    if mode == 'max':
        output = torch.topk(input_tensor, k, dim=0)[0]
    elif mode == 'min':
        output = torch.topk(input_tensor, k, dim=0, largest=False)[0]
    else:
        raise ValueError("Invalid mode. Choose 'max' or 'min'.")
    return output.to(torch.float32)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 4), torch.float32), 
        ("str", ),
        (int, )
    ],
    "outputs": [
        ((4, ), torch.float32)
    ]
}
