
import torch

def torch_nms_bfloat16_function(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on bounding boxes with bfloat16 precision.
    """
    boxes_bf16 = boxes.to(torch.bfloat16)
    scores_bf16 = scores.to(torch.bfloat16)

    # Calculate pairwise distances using Manhattan distance
    distances = torch.cdist(boxes_bf16, boxes_bf16, p=1)

    # Apply NMS based on distances and scores
    keep = torch.ones_like(scores_bf16, dtype=torch.bool)
    for i in range(len(scores_bf16)):
        if keep[i]:
            for j in range(i + 1, len(scores_bf16)):
                if keep[j] and distances[i, j] < iou_threshold:
                    if scores_bf16[i] < scores_bf16[j]:
                        keep[i] = False
                    else:
                        keep[j] = False

    return boxes_bf16[keep].to(torch.float32)

function_signature = {
    "name": "torch_nms_bfloat16_function",
    "inputs": [
        ((100, 4), torch.float32),  # Boxes (num_boxes, 4)
        ((100,), torch.float32),  # Scores (num_boxes)
        ((), torch.float32)  # IOU threshold
    ],
    "outputs": [
        ((None, 4), torch.float32),  # Selected boxes
    ]
}
