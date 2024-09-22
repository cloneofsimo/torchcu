
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(AttentionModule, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.W = nn.Linear(hidden_size, attention_size, bias=False)
        self.V = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        e = self.V(torch.tanh(self.W(x)))  # (batch_size, sequence_length, 1)
        alpha = F.softmax(e, dim=1)  # (batch_size, sequence_length, 1)
        context = torch.sum(alpha * x, dim=1)  # (batch_size, hidden_size)
        return context, alpha

class MyModel(nn.Module):
    def __init__(self, hidden_size, attention_size, temperature):
        super(MyModel, self).__init__()
        self.attention = AttentionModule(hidden_size, attention_size)
        self.temperature = temperature

    def forward(self, input_tensor, target_tensor):
        # Input tensor: (batch_size, seq_len, hidden_size)
        # Target tensor: (batch_size, seq_len, hidden_size)
        
        # Calculate attention context
        context, alpha = self.attention(input_tensor)
        
        # Apply log_softmax with temperature
        logits = F.log_softmax(context / self.temperature, dim=-1)  # (batch_size, hidden_size)
        
        # Calculate margin ranking loss
        loss = F.margin_ranking_loss(logits, torch.zeros_like(logits), target_tensor, margin=0.2)
        
        # Apply max filter to attention weights
        alpha_max = torch.max(alpha, dim=1)[0]  # (batch_size, 1)
        
        return loss, alpha_max

def my_function(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs attention, log_softmax with temperature, margin ranking loss, and max filtering.
    """
    model = MyModel(hidden_size=128, attention_size=64, temperature=2.0)
    loss, alpha_max = model(input_tensor.half(), target_tensor.half())
    return loss.float(), alpha_max.float()

function_signature = {
    "name": "my_function",
    "inputs": [
        ((16, 32, 128), torch.float16),
        ((16, 32, 128), torch.float16)
    ],
    "outputs": [
        ((1,), torch.float32),
        ((16,), torch.float32)
    ]
}
