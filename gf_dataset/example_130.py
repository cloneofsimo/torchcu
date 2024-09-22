
import torch
import torch.nn.functional as F

def torch_nms_prelu_int8_function(boxes: torch.Tensor, scores: torch.Tensor, prelu_weight: torch.Tensor) -> torch.Tensor:
    """
    Applies non-maximum suppression (NMS) to boxes with scores, followed by a PReLU activation.
    
    Args:
        boxes: Tensor of shape (N, 4) representing bounding boxes.
        scores: Tensor of shape (N,) representing scores for each box.
        prelu_weight: Tensor of shape (1,) representing the weight for the PReLU activation.
        
    Returns:
        Tensor of shape (M,) representing the indices of the selected boxes after NMS.
    """
    selected_indices = torch.ops.torchvision.nms(boxes.to(torch.float32), scores.to(torch.float32), 0.5)  # NMS
    
    selected_scores = scores[selected_indices].to(torch.int8)
    prelu_output = F.prelu(selected_scores.to(torch.float32), prelu_weight.to(torch.float32))  # PReLU
    
    return prelu_output.to(torch.int8)


function_signature = {
    "name": "torch_nms_prelu_int8_function",
    "inputs": [
        ((100, 4), torch.float32),
        ((100,), torch.float32),
        ((1,), torch.float32),
    ],
    "outputs": [
        ((None,), torch.int8),  # Output shape is dynamic, can't be known before runtime
    ]
}
