import torch
import torch.nn as nn
import torch.nn.functional as F

def simple_contrastive_model(
    input_ids: torch.Tensor, 
    pixel_values: torch.Tensor, 
    embedding_dim: int = 128
) -> torch.Tensor:
    # Text embedding
    text_embedding = nn.Linear(input_ids.shape[-1], embedding_dim)
    text_output = text_embedding(input_ids)
    text_output = F.normalize(text_output, p=2, dim=-1)

    # Image embedding
    image_embedding = nn.Linear(pixel_values.shape[-1], embedding_dim)
    image_output = image_embedding(pixel_values)
    image_output = F.normalize(image_output, p=2, dim=-1)

    # Cosine similarity
    similarity = F.cosine_similarity(text_output, image_output)

    return similarity



# function_signature
function_signature = {
    "name": "simple_contrastive_model",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4), torch.float32)]
}