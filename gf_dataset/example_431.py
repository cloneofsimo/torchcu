
import torch
import torch.nn.functional as F

def torch_cutmix_inplace_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor, 
                                   alpha: float = 1.0) -> torch.Tensor:
    """
    Performs CutMix augmentation inplace on the input tensor.
    """
    batch_size = input_tensor.size(0)
    
    # Generate random cutmix parameters
    lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size,)).to(input_tensor.device)
    
    # Calculate the size of the cutmix region
    cut_ratio = torch.sqrt(1 - lam).unsqueeze(1).unsqueeze(1)
    cut_size = (input_tensor.size(2) * cut_ratio).floor().long()
    
    # Generate random starting points for the cutmix region
    x1 = torch.randint(0, input_tensor.size(2) - cut_size[0, 0, 0] + 1, (batch_size,)).to(input_tensor.device)
    y1 = torch.randint(0, input_tensor.size(3) - cut_size[0, 0, 1] + 1, (batch_size,)).to(input_tensor.device)
    
    # Generate random indices for the cutmix operation
    perm = torch.randperm(batch_size)
    
    # Apply the cutmix operation
    for i in range(batch_size):
        x2 = x1[i] + cut_size[i, 0, 0]
        y2 = y1[i] + cut_size[i, 0, 1]
        
        input_tensor[i, :, x1[i]:x2, y1[i]:y2] = input_tensor[perm[i], :, x1[i]:x2, y1[i]:y2]
        target_tensor[i] = lam[i] * target_tensor[i] + (1 - lam[i]) * target_tensor[perm[i]]
    
    return input_tensor


function_signature = {
    "name": "torch_cutmix_inplace_function",
    "inputs": [
        ((3, 224, 224), torch.float32),
        ((3, ), torch.int64),
        (1.0, torch.float32),
    ],
    "outputs": [
        ((3, 224, 224), torch.float32),
    ]
}
