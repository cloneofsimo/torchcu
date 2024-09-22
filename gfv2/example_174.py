
import torch

def zero_crossing_rate_int8(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the zero crossing rate for each channel of the input tensor.
    The input tensor is assumed to be of shape (batch_size, channels, length).
    The output tensor is of shape (batch_size, channels).
    """
    assert input_tensor.ndim == 3, "Input tensor must have 3 dimensions (batch_size, channels, length)"

    # Convert to int8
    input_int8 = input_tensor.to(torch.int8)

    # Calculate the zero crossing rate for each channel
    zero_crossing_rate = torch.zeros((input_tensor.shape[0], input_tensor.shape[1]), dtype=torch.float32, device=input_tensor.device)
    for b in range(input_tensor.shape[0]):
        for c in range(input_tensor.shape[1]):
            for i in range(1, input_tensor.shape[2]):
                if (input_int8[b, c, i] * input_int8[b, c, i - 1]) < 0:
                    zero_crossing_rate[b, c] += 1

    # Normalize the zero crossing rate by the length of the input tensor
    zero_crossing_rate /= (input_tensor.shape[2] - 1)

    return zero_crossing_rate

function_signature = {
    "name": "zero_crossing_rate_int8",
    "inputs": [
        ((1, 10, 100), torch.float32),
    ],
    "outputs": [
        ((1, 10), torch.float32),
    ]
}
