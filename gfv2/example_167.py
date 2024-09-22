
import torch
import torch.nn as nn

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # Create coordinate grid
        grid_x = torch.arange(width, device=x.device).float()
        grid_y = torch.arange(height, device=x.device).float()
        grid_x = grid_x.repeat(height, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid_y = grid_y.repeat(width, 1).t().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # Concatenate coordinates with input
        x = torch.cat([x, grid_x / (width - 1), grid_y / (height - 1)], dim=1)
        return self.conv(x)

def cosine_embedding_loss_with_coordconv(input1: torch.Tensor, input2: torch.Tensor, 
                                            target: torch.Tensor, temperature: float = 1.0,
                                            reduction: str = "mean") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the cosine embedding loss with a CoordConv layer.

    Args:
        input1 (torch.Tensor): Input tensor 1 with shape [N, C, H, W].
        input2 (torch.Tensor): Input tensor 2 with shape [N, C, H, W].
        target (torch.Tensor): Target tensor with shape [N]. 
        temperature (float): Temperature scaling factor.
        reduction (str): Reduction method for the loss. Can be 'mean' or 'sum'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the loss and the logits. 
    """

    # Apply CoordConv to input tensors
    coordconv = CoordConv(input1.shape[1], input1.shape[1])
    input1 = coordconv(input1)
    input2 = coordconv(input2)

    # Calculate cosine similarity
    input1 = input1.view(input1.shape[0], -1)
    input2 = input2.view(input2.shape[0], -1)
    similarity = torch.nn.functional.cosine_similarity(input1, input2)

    # Apply temperature scaling
    similarity = similarity / temperature

    # Calculate the loss
    loss_fn = torch.nn.CosineEmbeddingLoss(reduction=reduction)
    loss = loss_fn(similarity, target, torch.ones_like(target))
    
    return loss, similarity
