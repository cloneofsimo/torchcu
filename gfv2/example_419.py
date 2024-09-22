
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.01, weight_decay: float = 0.001,
                 use_fp16: bool = False):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_fp16 = use_fp16

        # Define layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Enable mixed precision training if specified
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(input_tensor))
        x = self.linear2(x)
        return x

    def train_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.optimizer.zero_grad()

        if self.use_fp16:
            with torch.cuda.amp.autocast():
                output = self.forward(input_tensor)
                loss = F.mse_loss(output, target_tensor)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.forward(input_tensor)
            loss = F.mse_loss(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        return output, loss.item()

def my_function(input_tensor: torch.Tensor, hyperparameters: List[float], target_tensor: torch.Tensor = None) -> List[torch.Tensor]:
    """
    This function demonstrates a complex torch function with various concepts:
        - Hyperparameter optimization: The function accepts hyperparameters for model initialization.
        - BMM_out: Performs batch matrix multiplication with an output tensor.
        - Block_diag: Creates a block diagonal matrix.
        - Any: Checks if any element in a tensor is True.
        - Backward: Calculates gradients for training.
        - FP32 and FP16: Supports both floating-point precisions for training.
        - Forward: Executes the forward pass of the model.

    Args:
        input_tensor: Input tensor.
        hyperparameters: List of hyperparameters for model initialization.
        target_tensor: Target tensor for training (optional).

    Returns:
        A list of tensors:
            - Output tensor from the model's forward pass.
            - Loss tensor (only if target_tensor is provided).
    """

    # Initialize model with given hyperparameters
    learning_rate = hyperparameters[0]
    weight_decay = hyperparameters[1]
    use_fp16 = hyperparameters[2]
    model = MyModel(input_size=input_tensor.shape[1], hidden_size=64, output_size=32,
                    learning_rate=learning_rate, weight_decay=weight_decay, use_fp16=use_fp16)

    # Prepare tensors for block_diag operation
    identity_matrix = torch.eye(input_tensor.shape[1]).unsqueeze(0)
    block_diagonal = torch.block_diag(*[identity_matrix for _ in range(input_tensor.shape[0])])

    # Perform batch matrix multiplication
    output = torch.bmm(input_tensor.unsqueeze(1), block_diagonal.unsqueeze(0))

    # Forward pass and potentially backward pass
    if target_tensor is not None:
        # Train the model using provided hyperparameters
        output, loss = model.train_step(output, target_tensor)
        # Check if any element in the loss tensor is greater than 1.0
        is_loss_high = (loss > 1.0).any()
        # Return output and loss tensors
        return [output, torch.tensor([loss])]
    else:
        # Return output tensor only
        return [output]

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 4), torch.float32),
        ([0.01, 0.001, False], torch.float32),
        ((10, 32), torch.float32)
    ],
    "outputs": [
        ((10, 32), torch.float32),
        ((1,), torch.float32),
    ]
}
