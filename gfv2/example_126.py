
import torch

class LocalAttention(torch.nn.Module):
    def __init__(self, d_model, window_size, causal=False):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.causal = causal

        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Local attention
        start_idx = torch.arange(T, device=x.device)[:, None] - (self.window_size - 1)
        end_idx = torch.arange(T, device=x.device)[:, None] + self.window_size
        
        # Ensure causal masking if required
        if self.causal:
            end_idx = torch.clip(end_idx, max=start_idx + self.window_size)
        
        start_idx = torch.clamp(start_idx, min=0)
        end_idx = torch.clamp(end_idx, max=T)
        
        k = torch.gather(k, dim=1, index=start_idx[:, :, None, None].repeat(1, 1, C, 1))
        v = torch.gather(v, dim=1, index=start_idx[:, :, None, None].repeat(1, 1, C, 1))
        
        # Compute attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        
        # Causal masking
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1) == 1
            attn = torch.where(mask[:, :, None, None].repeat(1, 1, C, C), -float('inf'), attn)
        
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn, v)
        output = self.proj(output.permute(1, 0, 2, 3)).view(B, T, C)
        
        return output

function_signature = {
    "name": "local_attention_forward",
    "inputs": [
        ((1, 10, 512), torch.float32),
    ],
    "outputs": [
        ((1, 10, 512), torch.float32),
    ]
}
