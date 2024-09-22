
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Calculate weighted sum of values
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MyModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, dropout=0.1):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads

        # Embedding Layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Multi-Head Attention
        self.attention = ScaledDotProductAttention(d_model, dropout)

        # Feedforward Network
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )

        # Squeeze-and-Excitation (SE) Module
        self.se = SELayer(d_model)

        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # Input Embedding
        x = self.embedding(input_tensor)

        # Multi-Head Attention
        q = k = v = self.norm(x)
        output, attn_weights = self.attention(q, k, v)

        # Feedforward Network
        output = self.fc(output)

        # SE Module
        output = self.se(output)

        # Add Residual Connection
        output = self.dropout(output) + x

        return output, attn_weights

def model_function(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function takes an input tensor and performs multi-head attention, feedforward network, SE module, and 
    layer normalization. It returns the output tensor and attention weights.
    """
    model = MyModel(input_dim=10, d_model=64, num_heads=8, dropout=0.1).to(torch.bfloat16)
    output, attn_weights = model(input_tensor)
    return output.to(torch.float32), attn_weights.to(torch.float32)

function_signature = {
    "name": "model_function",
    "inputs": [
        ((1, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 64, 10), torch.float32),
        ((1, 10, 10), torch.float32),
    ]
}
