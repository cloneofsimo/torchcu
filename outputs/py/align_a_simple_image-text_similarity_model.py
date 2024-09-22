import torch
import torch.nn as nn
import torch.nn.functional as F

def image_text_similarity(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """
    This function calculates the similarity between image features and text features.
    
    Args:
    image_features (torch.Tensor): A tensor of shape (batch_size, image_feature_dim) representing the features of images.
    text_features (torch.Tensor): A tensor of shape (batch_size, text_feature_dim) representing the features of text.
    
    Returns:
    torch.Tensor: A tensor of shape (batch_size, batch_size) representing the similarity between image features and text features.
    """
    
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    
    # Calculate the similarity
    similarity = torch.matmul(image_features, text_features.T)
    
    return similarity



# function_signature
function_signature = {
    "name": "image_text_similarity",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [((4, 4), torch.float32)]
}