
import torch
import torch.nn.functional as F

def supervised_contrastive_loss_fp16(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """
    Calculates the supervised contrastive loss.
    """
    anchor_fp16 = anchor.to(torch.float16)
    positive_fp16 = positive.to(torch.float16)
    negative_fp16 = negative.to(torch.float16)

    similarity_ap = F.cosine_similarity(anchor_fp16, positive_fp16, dim=1)
    similarity_an = F.cosine_similarity(anchor_fp16, negative_fp16, dim=1)

    loss = torch.mean(torch.nn.functional.relu(1.0 - similarity_ap + similarity_an))
    return loss.to(torch.float32)

function_signature = {
    "name": "supervised_contrastive_loss_fp16",
    "inputs": [
        ((10, 128), torch.float32),
        ((10, 128), torch.float32),
        ((10, 128), torch.float32)
    ],
    "outputs": [
        ((), torch.float32),
    ]
}
