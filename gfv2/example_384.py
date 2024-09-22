
import torch
import torch.nn.functional as F

def complex_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, input_tensor3: torch.Tensor, 
                    input_tensor4: torch.Tensor, input_tensor5: torch.Tensor) -> torch.Tensor:
    """
    Performs a complex series of operations on input tensors, 
    including multi-margin loss, sum, max pooling, broadcasting, bfloat16 and int8 operations.
    """
    # 1. Multi-margin loss
    margin_loss = F.multi_margin_loss(input_tensor1.to(torch.float32), torch.tensor([0, 1, 2]), p=1)

    # 2. Sum
    sum_tensor = torch.sum(input_tensor2)

    # 3. Max pooling
    max_pooled_tensor = F.max_pool2d(input_tensor3, kernel_size=3, stride=2)

    # 4. Broadcasting
    broadcasted_tensor = input_tensor4.unsqueeze(0) + input_tensor5.unsqueeze(1)

    # 5. bfloat16 and int8 operations
    bf16_tensor = input_tensor1.to(torch.bfloat16)
    int8_tensor = input_tensor2.to(torch.int8)

    # 6. Inplace operations
    input_tensor3.add_(1)

    # 7. Combine results
    output_tensor = margin_loss + sum_tensor + max_pooled_tensor + broadcasted_tensor + bf16_tensor + int8_tensor
    
    return output_tensor

function_signature = {
    "name": "complex_function",
    "inputs": [
        ((1,), torch.float32),
        ((1,), torch.float32),
        ((2, 2, 4, 4), torch.float32),
        ((2, 2), torch.float32),
        ((2, 2), torch.float32),
    ],
    "outputs": [
        ((2, 2, 4, 4), torch.float32),
    ]
}
