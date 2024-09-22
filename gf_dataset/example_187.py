
import torch
import torch.nn as nn
import torch.nn.functional as F

def knowledge_distillation_loss_bf16(student_output: torch.Tensor, teacher_output: torch.Tensor, 
                                    image_gradient: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes the knowledge distillation loss using bfloat16 for efficiency.
    
    Args:
        student_output (torch.Tensor): Output of the student model.
        teacher_output (torch.Tensor): Output of the teacher model.
        image_gradient (torch.Tensor): Image gradient used to weight the loss.
        temperature (float, optional): Temperature scaling factor. Defaults to 1.0.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    """
    student_output_bf16 = student_output.to(torch.bfloat16)
    teacher_output_bf16 = teacher_output.to(torch.bfloat16)
    image_gradient_bf16 = image_gradient.to(torch.bfloat16)

    # Softmax with temperature scaling
    student_probs = F.softmax(student_output_bf16 / temperature, dim=1)
    teacher_probs = F.softmax(teacher_output_bf16 / temperature, dim=1)

    # Kullback-Leibler divergence with weighted average
    kl_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='none')
    weighted_kl_loss = (kl_loss * image_gradient_bf16).mean()

    return weighted_kl_loss.to(torch.float32)

function_signature = {
    "name": "knowledge_distillation_loss_bf16",
    "inputs": [
        ((1, 1000), torch.float32),
        ((1, 1000), torch.float32),
        ((3, 224, 224), torch.float32),
    ],
    "outputs": [
        ((1,), torch.float32),
    ]
}
