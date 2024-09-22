
import torch
from torch import nn
from torch.nn import functional as F
from cutlass import *

def torch_rel_pos_median_fp16(input_tensor: torch.Tensor, rel_pos_emb: torch.Tensor,
                               head_num: int, device: torch.device) -> torch.Tensor:
    """
    Computes relative positional encoding with median aggregation using fp16
    """
    # Cast to fp16
    input_tensor_fp16 = input_tensor.to(torch.float16)
    rel_pos_emb_fp16 = rel_pos_emb.to(torch.float16)

    # Reshape for efficient computation
    batch_size, seq_len, embed_dim = input_tensor.shape
    input_tensor_fp16 = input_tensor_fp16.view(batch_size, seq_len, head_num, embed_dim // head_num)
    rel_pos_emb_fp16 = rel_pos_emb_fp16.view(seq_len, seq_len, head_num, embed_dim // head_num)

    # Compute dot product with relative positional embeddings
    attn_weights = torch.einsum('bhid,ijhd->bhj', input_tensor_fp16, rel_pos_emb_fp16)

    # Median aggregation over the relative positions
    median_attn_weights = torch.median(attn_weights, dim=2, keepdim=True).values

    # Reshape and return
    median_attn_weights = median_attn_weights.view(batch_size, seq_len, embed_dim)
    return median_attn_weights.to(torch.float32)

function_signature = {
    "name": "torch_rel_pos_median_fp16",
    "inputs": [
        ((16, 32, 512), torch.float32),
        ((32, 32, 128), torch.float32),
        (16, torch.int32),
        (torch.device,),
    ],
    "outputs": [
        ((16, 32, 512), torch.float32),
    ]
}
