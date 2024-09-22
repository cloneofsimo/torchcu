
import torch
import torch.nn.functional as F

def fused_ln_dropout_fp16(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """
    Performs fused layer normalization, dropout, and activation in fp16.
    """
    input_fp16 = input_tensor.to(torch.float16)
    weight_fp16 = weight.to(torch.float16)
    bias_fp16 = bias.to(torch.float16)
    
    # Fused layer normalization
    output = F.layer_norm(input_fp16, weight_fp16, bias_fp16)
    
    # Dropout
    output = F.dropout(output, p=dropout_p, training=True)
    
    # Activation (ReLU in this example)
    output = F.relu(output)
    
    return output.to(torch.float32)

function_signature = {
    "name": "fused_ln_dropout_fp16",
    "inputs": [
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
        (0.5, torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
