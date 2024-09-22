
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(embed_dim, head_dim * num_heads)
        self.W_k = nn.Linear(embed_dim, head_dim * num_heads)
        self.W_v = nn.Linear(embed_dim, head_dim * num_heads)
        self.W_o = nn.Linear(head_dim * num_heads, embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, x, mask):
        # Shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, head_dim * num_heads)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # (batch_size, num_heads, seq_len, seq_len)
        attention = torch.bmm(q, k.transpose(2, 3)) * self.scale
        attention = torch.where(mask.unsqueeze(1).unsqueeze(1), attention, -1e9)
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # (batch_size, num_heads, seq_len, head_dim)
        context = torch.bmm(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        # (batch_size, seq_len, embed_dim)
        output = self.W_o(context)
        output = torch.relu(output)
        return output

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs a simple linear transformation, applies tanh activation, 
    pads the output with constant values, multiplies it by weight, 
    and then performs masked attention.
    """
    # (batch_size, seq_len, embed_dim)
    output = torch.matmul(input_tensor, weight.t())
    output = torch.tanh(output)

    # (batch_size, seq_len + 2, embed_dim)
    output = torch.nn.functional.pad(output, (0, 0, 1, 1), "constant", 0.0)

    # (batch_size, seq_len + 2, embed_dim)
    output = output * weight

    # (batch_size, seq_len + 2, embed_dim)
    output = MyModule(embed_dim=output.shape[-1], head_dim=16, num_heads=4, dropout=0.1)(output, mask)

    return output

function_signature = {
    "name": "my_function",
    "inputs": [
        ((10, 12, 32), torch.float32),
        ((10, 12, 32), torch.float32),
        ((10, 12), torch.bool)
    ],
    "outputs": [
        ((10, 12, 32), torch.float32)
    ]
}
