
import torch
import torch.nn.functional as F
from torch import nn

class DetrTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
                                             num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation),
                                             num_decoder_layers)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Args:
            src: (batch_size, seq_len_src, d_model)
            tgt: (batch_size, seq_len_tgt, d_model)
            src_mask: (batch_size, seq_len_src)
            tgt_mask: (batch_size, seq_len_tgt)
            memory_mask: (batch_size, seq_len_tgt, seq_len_src)

        Returns:
            output: (batch_size, seq_len_tgt, d_model)
        """
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output


def detrs_function(input_tensor: torch.Tensor, query_embed: torch.Tensor, mask_tensor: torch.Tensor,
                   detr_transformer: DetrTransformer) -> torch.Tensor:
    """
    Performs DETR's transformer operations using bfloat16.

    Args:
        input_tensor: (batch_size, seq_len_src, d_model)
        query_embed: (seq_len_tgt, d_model)
        mask_tensor: (batch_size, seq_len_src)
        detr_transformer: A DETR transformer instance

    Returns:
        output: (batch_size, seq_len_tgt, d_model)
    """
    # Convert inputs to bfloat16 for efficiency
    input_tensor_bf16 = input_tensor.to(torch.bfloat16)
    query_embed_bf16 = query_embed.to(torch.bfloat16)
    mask_tensor_bf16 = mask_tensor.to(torch.bfloat16)

    # Expand query_embed for batch processing
    query_embed_expanded = query_embed_bf16.unsqueeze(0).expand(input_tensor_bf16.size(0), -1, -1)

    # Perform Transformer operation
    output = detr_transformer(input_tensor_bf16, query_embed_expanded, mask_tensor_bf16, None, None)

    # Convert back to float32
    return output.to(torch.float32)


function_signature = {
    "name": "detrs_function",
    "inputs": [
        ((16, 100, 256), torch.float32),
        ((100, 256), torch.float32),
        ((16, 100), torch.bool),
    ],
    "outputs": [
        ((16, 100, 256), torch.float32),
    ]
}
