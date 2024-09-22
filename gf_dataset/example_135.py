
import torch

def torch_nms_fp32(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.
    Returns the indices of the selected boxes.

    Args:
        boxes: (N, 4) tensor of bounding boxes in (x1, y1, x2, y2) format.
        scores: (N,) tensor of scores for each bounding box.
        iou_threshold: IoU threshold for suppression.

    Returns:
        A (K,) tensor of indices of the selected bounding boxes.
    """
    keep = torch.nms(boxes, scores, iou_threshold)
    return keep

function_signature = {
    "name": "torch_nms_fp32",
    "inputs": [
        ((100, 4), torch.float32),
        ((100,), torch.float32),
        ((), torch.float32)
    ],
    "outputs": [
        ((100,), torch.int64),
    ]
}
