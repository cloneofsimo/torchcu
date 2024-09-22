
import torch
import torch.nn as nn

def teacher_student_loss_fp16(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Calculates the teacher-student loss using adaptive log softmax and fp16 for efficiency.
    
    Args:
        student_output (torch.Tensor): Output of the student model (B x C).
        teacher_output (torch.Tensor): Output of the teacher model (B x C).
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: The teacher-student loss value.
    """
    student_output_fp16 = student_output.to(torch.float16)
    teacher_output_fp16 = teacher_output.to(torch.float16)

    student_log_probs = nn.functional.adaptive_log_softmax(student_output_fp16, dim=1)
    teacher_probs = nn.functional.softmax(teacher_output_fp16 / temperature, dim=1)

    loss = -(teacher_probs * student_log_probs).sum(dim=1).mean()

    return loss.to(torch.float32)  # Return loss in fp32

function_signature = {
    "name": "teacher_student_loss_fp16",
    "inputs": [
        ((16, 10), torch.float32),  # Example input shape
        ((16, 10), torch.float32),  # Example input shape
        (1.0, torch.float32)
    ],
    "outputs": [
        ((1,), torch.float32)
    ]
}

