
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def teacher_student_training(teacher_output: torch.Tensor, student_output: torch.Tensor, 
                             teacher_target: torch.Tensor, student_target: torch.Tensor,
                             alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    """
    Implements a teacher-student training loss combining knowledge distillation and supervised learning.

    Args:
        teacher_output: Output tensor from the teacher model (B, C, H, W)
        student_output: Output tensor from the student model (B, C, H, W)
        teacher_target: Target tensor for the teacher model (B, C, H, W)
        student_target: Target tensor for the student model (B, C, H, W)
        alpha: Weight for knowledge distillation loss
        beta: Weight for supervised learning loss

    Returns:
        Combined loss tensor
    """
    # Knowledge distillation loss
    distillation_loss = F.mse_loss(student_output, teacher_output)

    # Supervised learning loss
    supervised_loss = F.mse_loss(student_output, student_target)

    # Combined loss
    loss = alpha * distillation_loss + beta * supervised_loss
    return loss

function_signature = {
    "name": "teacher_student_training",
    "inputs": [
        ((4, 4, 4, 4), torch.float32),
        ((4, 4, 4, 4), torch.float32),
        ((4, 4, 4, 4), torch.float32),
        ((4, 4, 4, 4), torch.float32),
        (0.5, torch.float32),
        (0.5, torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}

