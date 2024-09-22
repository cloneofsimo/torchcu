
import torch
import torch.nn.functional as F

def torch_gumbel_softmax_dropout_function(input_tensor: torch.Tensor, weight: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Applies Gumbel-Softmax, dropout, and a linear transformation to the input tensor.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    
    # Gumbel-Softmax
    gumbel_output = F.gumbel_softmax(input_bf16, tau=1.0, hard=True)
    
    # Dropout
    gumbel_output = F.dropout(gumbel_output, p=dropout_p, training=True, inplace=True)
    
    # Linear Transformation
    output_bf16 = torch.matmul(gumbel_output, weight_bf16.t())
    
    # Power operation
    output_bf16 = output_bf16.pow(2)
    
    # Addcmul
    output_bf16 = output_bf16.addcmul(input_bf16, weight_bf16, value=0.5)

    return output_bf16.to(torch.float32)

function_signature = {
    "name": "torch_gumbel_softmax_dropout_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        (0.5, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
