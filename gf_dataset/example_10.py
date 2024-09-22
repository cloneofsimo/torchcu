
import torch
import torch.nn as nn

class TripletLossNet(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossNet, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Computes the Triplet Margin Loss.

        Args:
            anchor (torch.Tensor): The anchor tensor.
            positive (torch.Tensor): The positive tensor.
            negative (torch.Tensor): The negative tensor.

        Returns:
            torch.Tensor: The triplet loss value.
        """
        # Calculate distances
        dist_ap = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
        dist_an = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1))

        # Calculate the loss
        loss = torch.nn.functional.triplet_margin_loss(
            anchor=dist_ap, 
            positive=dist_an, 
            margin=self.margin, 
            reduction='mean' 
        )
        return loss

function_signature = {
    "name": "triplet_loss_net",
    "inputs": [
        ((10, 3), torch.float32),
        ((10, 3), torch.float32),
        ((10, 3), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
