
import torch
import torch.nn as nn

class DETRTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Encode the source sequence
        memory = self.encoder(src, mask=src_mask)

        # Decode the target sequence
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return output

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        output, attn = self.attention(
            query, key, value,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        output = self.dropout(output)
        return output, attn

def detr_transformer_with_cross_attention(input_tensor: torch.Tensor, memory: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a DETR transformer with cross-attention for object detection.

    Args:
        input_tensor (torch.Tensor): The input tensor to the transformer, shape (B, N, d_model)
        memory (torch.Tensor): The encoded memory from the encoder, shape (B, M, d_model)
        query_mask (torch.Tensor): Mask for the queries, shape (B, N)

    Returns:
        torch.Tensor: The output of the transformer, shape (B, N, d_model)
    """

    # Define transformer parameters
    d_model = 256
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1

    # Initialize DETR Transformer
    detr_transformer = DETRTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    # Initialize Cross-Attention module
    cross_attention = CrossAttention(d_model, nhead, dropout)

    # Apply DETR transformer
    output = detr_transformer(input_tensor, input_tensor, src_mask=query_mask, tgt_mask=query_mask)

    # Apply Cross-Attention
    output, attn = cross_attention(output, memory, memory, key_padding_mask=~query_mask)

    # Return the output
    return output.to(torch.bfloat16)


function_signature = {
    "name": "detr_transformer_with_cross_attention",
    "inputs": [
        ((100, 256), torch.float32),  # input_tensor
        ((100, 256), torch.float32),  # memory
        ((100,), torch.bool)  # query_mask
    ],
    "outputs": [
        ((100, 256), torch.bfloat16) # output
    ]
}
