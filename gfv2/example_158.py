
import torch
import torch.nn as nn
import torch.nn.functional as F

class DetrTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout), num_decoder_layers)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, src, tgt, memory_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, seq_len_src, d_model)
            tgt: (batch_size, seq_len_tgt, d_model)
            memory_mask: (batch_size, seq_len_tgt, seq_len_src)
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt)

        Returns:
            output: (batch_size, seq_len_tgt, d_model)
        """
        src = self.encoder(src, mask=memory_mask)
        output = self.decoder(tgt, src, tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.linear(output)
        return output

def detr_transformer_bf16_function(image_features: torch.Tensor, queries: torch.Tensor, memory_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs a DETR Transformer with bfloat16 precision.

    Args:
        image_features: (batch_size, seq_len_src, d_model)
        queries: (batch_size, seq_len_tgt, d_model)
        memory_mask: (batch_size, seq_len_tgt, seq_len_src)
        tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt)

    Returns:
        output: (batch_size, seq_len_tgt, d_model)
    """
    image_features_bf16 = image_features.to(torch.bfloat16)
    queries_bf16 = queries.to(torch.bfloat16)
    memory_mask_bf16 = memory_mask.to(torch.bfloat16)
    tgt_mask_bf16 = tgt_mask.to(torch.bfloat16)

    transformer = DetrTransformer().to(torch.bfloat16)
    output_bf16 = transformer(image_features_bf16, queries_bf16, memory_mask=memory_mask_bf16, tgt_mask=tgt_mask_bf16)
    return output_bf16.to(torch.float32)

function_signature = {
    "name": "detr_transformer_bf16_function",
    "inputs": [
        ((1, 100, 256), torch.float32),
        ((1, 100, 256), torch.float32),
        ((1, 100, 100), torch.float32),
        ((1, 100, 100), torch.float32)
    ],
    "outputs": [
        ((1, 100, 256), torch.float32),
    ]
}
