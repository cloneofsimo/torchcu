
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim, margin=1.0):
        super(TripletNetwork, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)

        distance_ap = F.pairwise_distance(anchor_embedding, positive_embedding)
        distance_an = F.pairwise_distance(anchor_embedding, negative_embedding)

        loss = torch.clamp(distance_ap - distance_an + self.margin, min=0.0)
        return loss, anchor_embedding, positive_embedding, negative_embedding

def torch_triplet_loss_int8_function(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    """
    Computes the triplet loss using int8 quantization for embeddings.
    """
    model = TripletNetwork(embedding_dim=128, margin=margin)
    model.embedding_net.to(torch.int8)
    loss, _, _, _ = model(anchor.to(torch.int8), positive.to(torch.int8), negative.to(torch.int8))
    return loss.to(torch.float32)

function_signature = {
    "name": "torch_triplet_loss_int8_function",
    "inputs": [
        ((1, 1, 28, 28), torch.float32),
        ((1, 1, 28, 28), torch.float32),
        ((1, 1, 28, 28), torch.float32),
        (1, torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}

