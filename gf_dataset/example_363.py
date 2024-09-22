
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def transformer_layer_nms_fp16(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, 
                                 head_size: int, nms_top_k: int, nms_threshold: float) -> torch.Tensor:
    """
    A transformer layer with NMS applied to the output.

    Args:
        query: Query tensor (batch_size, seq_len_q, head_size)
        key: Key tensor (batch_size, seq_len_k, head_size)
        value: Value tensor (batch_size, seq_len_k, head_size)
        mask: Attention mask (batch_size, seq_len_q, seq_len_k)
        head_size: Size of each attention head
        nms_top_k: Number of top-scoring elements to keep after NMS
        nms_threshold: NMS threshold

    Returns:
        Output tensor (batch_size, seq_len_q, head_size) after NMS
    """

    with autocast():
        attention = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (head_size ** 0.5) + mask, dim=-1)
        output = torch.matmul(attention, value)  
        output = output.float() 

    # Apply NMS along the sequence dimension (seq_len_q)
    for i in range(output.size(0)): 
        scores = output[i, :, 0] # Assuming head_size is 1 in this example 
        keep_indices = torch.argsort(scores, descending=True)[:nms_top_k]
        output[i] = torch.masked_select(output[i], keep_indices) 
        output[i] = F.pad(output[i], (0, 0, 0, nms_top_k - output[i].shape[0]), "constant", 0) 
    return output

function_signature = {
    "name": "transformer_layer_nms_fp16",
    "inputs": [
        ((2, 10, 1), torch.float32), # query
        ((2, 10, 1), torch.float32), # key
        ((2, 10, 1), torch.float32), # value
        ((2, 10, 10), torch.float32), # mask
        (1, torch.int32), # head_size
        (1, torch.int32), # nms_top_k
        (1, torch.float32)  # nms_threshold
    ],
    "outputs": [
        ((2, 10, 1), torch.float32)
    ]
}
