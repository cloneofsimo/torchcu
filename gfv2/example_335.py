
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def sparse_int8_padding_function(inputs: list[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """
    Performs padding on a list of sparse tensors, converts them to int8, and concatenates them.

    Args:
        inputs: A list of sparse tensors to be padded.
        padding_value: The value to use for padding.

    Returns:
        A single concatenated tensor of int8 dtype.
    """
    # Pad the input tensors to the same length
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)

    # Convert the padded tensor to int8
    padded_inputs_int8 = padded_inputs.to(torch.int8)

    # Concatenate the padded tensors
    output = torch.cat(padded_inputs_int8, dim=1)

    return output

function_signature = {
    "name": "sparse_int8_padding_function",
    "inputs": [
        [((10,), torch.int64), ((12,), torch.int64)],
        ((), torch.int64)
    ],
    "outputs": [
        ((10, 22), torch.int8),
    ]
}
