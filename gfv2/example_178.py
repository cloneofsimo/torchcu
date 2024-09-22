
import torch

def nms_exponential_bf16(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes with exponential scoring.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 4) representing bounding boxes, where each row is
                             [x1, y1, x2, y2].
        scores (torch.Tensor): A tensor of shape (N,) representing the scores of each bounding box.
        iou_threshold (float): The IoU threshold for suppression.

    Returns:
        torch.Tensor: A tensor of indices of the kept bounding boxes.
    """
    # Convert to bfloat16
    boxes_bf16 = boxes.to(torch.bfloat16)
    scores_bf16 = scores.to(torch.bfloat16)
    
    # Apply exponential scoring
    scores_bf16 = torch.exp(scores_bf16)
    
    # Perform NMS
    keep_bf16 = torch.nms(boxes_bf16, scores_bf16, iou_threshold)
    
    # Convert back to float32 and return
    return keep_bf16.to(torch.float32)


function_signature = {
    "name": "nms_exponential_bf16",
    "inputs": [
        ((10, 4), torch.float32),
        ((10,), torch.float32),
        ((), torch.float32),
    ],
    "outputs": [
        ((10,), torch.float32),
    ]
}
