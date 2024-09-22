import torch

def tokenize_input(
    input_tensor: torch.Tensor, 
    vocab: torch.Tensor, 
    merges: torch.Tensor, 
    max_length: int = 512, 
    pad_token: int = 0, 
    sep_token: int = 1
) -> torch.Tensor:
    """
    Tokenize input tensor using vocabulary and merges.

    Args:
    input_tensor (torch.Tensor): Input tensor to be tokenized.
    vocab (torch.Tensor): Vocabulary tensor.
    merges (torch.Tensor): Merges tensor.
    max_length (int, optional): Maximum length of the output tensor. Defaults to 512.
    pad_token (int, optional): Pad token. Defaults to 0.
    sep_token (int, optional): Separator token. Defaults to 1.

    Returns:
    torch.Tensor: Tokenized input tensor.
    """

    # Initialize output tensor with zeros
    output_tensor = torch.zeros((input_tensor.shape[0], max_length), dtype=torch.long)

    # Iterate over input tensor
    for i, input_id in enumerate(input_tensor):
        # Initialize tokenized input with bos token
        tokenized_input = [sep_token]

        # Iterate over input id
        for j, id in enumerate(input_id):
            # If id is not zero, add it to tokenized input
            if id != 0:
                tokenized_input.append(id)

        # Add eos token to tokenized input
        tokenized_input.append(sep_token)

        # Pad tokenized input to max length
        tokenized_input += [pad_token] * (max_length - len(tokenized_input))

        # Convert tokenized input to tensor and add to output tensor
        output_tensor[i] = torch.tensor(tokenized_input[:max_length])

    return output_tensor



# function_signature
function_signature = {
    "name": "tokenize_input",
    "inputs": [
        ((4, 4), torch.long),
        ((4, 4), torch.long),
        ((4, 4), torch.long)
    ],
    "outputs": [((4, 512), torch.int64)]
}