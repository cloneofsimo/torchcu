
import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def teacher_student_training(image: torch.Tensor, teacher_net: TeacherNet, student_net: StudentNet, 
                             grid: torch.Tensor, teacher_output: torch.Tensor, 
                             learning_rate: float = 0.001) -> torch.Tensor:
    """
    Performs one step of teacher-student training.

    Args:
        image: Input image tensor (batch_size, 3, 32, 32).
        teacher_net: Trained teacher network.
        student_net: Student network to be trained.
        grid: Grid tensor for grid sampling.
        teacher_output: Output of the teacher network.
        learning_rate: Learning rate for the student network.

    Returns:
        The output of the student network.
    """
    student_output = student_net(image)
    loss = F.mse_loss(student_output, teacher_output)

    student_net.zero_grad()
    loss.backward()

    for name, param in student_net.named_parameters():
        if param.grad is not None:
            param.data -= learning_rate * param.grad.data
    
    return student_output

function_signature = {
    "name": "teacher_student_training",
    "inputs": [
        ((1, 3, 32, 32), torch.float32),
        (None, TeacherNet),
        (None, StudentNet),
        ((1, 1, 3, 3), torch.float32),
        ((1, 10), torch.float32)
    ],
    "outputs": [
        ((1, 10), torch.float32)
    ]
}
