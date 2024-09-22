
import torch
import torch.nn.functional as F

class MyModule(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.depthwise_conv = torch.nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=3,
            groups=self.d_model,
            padding=1,
            bias=False
        )
        self.pe_proj = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.qkv_proj = torch.nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        # (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        # (batch_size, d_model, seq_len) -> (batch_size, d_model, seq_len, 1)
        x = x.unsqueeze(-1)

        # Depthwise convolution
        x = self.depthwise_conv(x)
        # (batch_size, d_model, seq_len, 1) -> (batch_size, seq_len, d_model)
        x = x.squeeze(-1).transpose(1, 2)

        # Learned Positional Encoding
        pe = self.pe_proj(x)
        x = x + pe

        # Linear Attention
        qkv = self.qkv_proj(x)
        # (batch_size, seq_len, 3*d_model) -> (batch_size, seq_len, 3, d_model)
        qkv = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.d_model)
        # (batch_size, seq_len, 3, d_model) -> (batch_size, 3, seq_len, d_model)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv, 1, dim=1)
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.squeeze(1).view(q.shape[0], self.num_heads, q.shape[1], self.head_dim).to(torch.bfloat16)
        k = k.squeeze(1).view(k.shape[0], self.num_heads, k.shape[1], self.head_dim).to(torch.bfloat16)
        v = v.squeeze(1).view(v.shape[0], self.num_heads, v.shape[1], self.head_dim).to(torch.bfloat16)
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, head_dim, seq_len)
        k = k.transpose(2, 3)
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention = torch.matmul(q, k)
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention = attention / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.bfloat16))
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attention = F.softmax(attention, dim=-1)
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attention = torch.matmul(attention, v)
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        attention = attention.permute(0, 2, 1, 3)
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, d_model)
        attention = attention.reshape(attention.shape[0], attention.shape[1], -1)
        attention = attention.to(torch.float32)

        # Output Projection
        output = self.out_proj(attention)
        output = self.dropout(output)

        # (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        output = output.transpose(1, 2)
        # (batch_size, d_model, seq_len) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2)
        return output

def my_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    model = MyModule(d_model=512, num_heads=8)
    return model(input_tensor)

function_signature = {
    "name": "my_function",
    "inputs": [
        ((4, 16, 512), torch.float32),
        ((512, 512), torch.float32)
    ],
    "outputs": [
        ((4, 16, 512), torch.float32),
    ]
}
