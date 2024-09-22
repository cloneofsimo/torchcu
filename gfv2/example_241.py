
import torch

def complex_tensor_operation(input_tensor: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
    """
    Performs a series of complex tensor operations using bfloat16 and fp16 precision.
    """

    # Reshape the input tensor to a 2D matrix
    input_tensor = input_tensor.reshape(-1, 4)

    # Convert the input tensor to bfloat16
    input_tensor = input_tensor.to(torch.bfloat16)

    # Unsqueeze the target values to create a 2D matrix
    target_values = target_values.unsqueeze(dim=0).to(torch.bfloat16)

    # Check if values in the input tensor are present in the target values
    mask = torch.isin(input_tensor, target_values)

    # Convert the mask to fp16
    mask = mask.to(torch.float16)

    # Perform a simple matrix multiplication using fp16
    output = torch.matmul(input_tensor.float(), mask.float())

    # Convert the output back to float32
    output = output.to(torch.float32)

    return output

function_signature = {
    "name": "complex_tensor_operation",
    "inputs": [
        ((16,), torch.float32),
        ((2,), torch.float32)
    ],
    "outputs": [
        ((4,), torch.float32)
    ]
}
