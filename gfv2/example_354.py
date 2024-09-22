
import torch

def transformer_decoder_layer(
    tgt: torch.Tensor, 
    memory: torch.Tensor,
    tgt_mask: torch.Tensor, 
    tgt_key_padding_mask: torch.Tensor,
    memory_key_padding_mask: torch.Tensor, 
    q_proj_weight: torch.Tensor, 
    k_proj_weight: torch.Tensor, 
    v_proj_weight: torch.Tensor, 
    in_proj_weight: torch.Tensor, 
    in_proj_bias: torch.Tensor, 
    linear1_weight: torch.Tensor, 
    linear1_bias: torch.Tensor, 
    linear2_weight: torch.Tensor, 
    linear2_bias: torch.Tensor, 
    dropout1: float, 
    dropout2: float, 
    dropout3: float, 
    activation: str, 
    norm1_weight: torch.Tensor, 
    norm1_bias: torch.Tensor, 
    norm2_weight: torch.Tensor, 
    norm2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transformer decoder layer.

    Args:
        tgt: target sequence
        memory: encoder output
        tgt_mask: target mask
        tgt_key_padding_mask: target key padding mask
        memory_key_padding_mask: memory key padding mask
        q_proj_weight: query projection weight
        k_proj_weight: key projection weight
        v_proj_weight: value projection weight
        in_proj_weight: input projection weight
        in_proj_bias: input projection bias
        linear1_weight: first linear layer weight
        linear1_bias: first linear layer bias
        linear2_weight: second linear layer weight
        linear2_bias: second linear layer bias
        dropout1: first dropout layer probability
        dropout2: second dropout layer probability
        dropout3: third dropout layer probability
        activation: activation function
        norm1_weight: first layer normalization weight
        norm1_bias: first layer normalization bias
        norm2_weight: second layer normalization weight
        norm2_bias: second layer normalization bias

    Returns:
        tgt: output of the decoder layer
        attn_weights: attention weights
    """
    tgt2 = torch.nn.functional.linear(tgt, in_proj_weight, in_proj_bias)
    tgt2 = torch.nn.functional.relu(tgt2)
    tgt2 = torch.nn.functional.dropout(tgt2, p=dropout1, training=False)
    tgt2 = torch.nn.functional.layer_norm(tgt2, tgt2.shape[-1:], weight=norm1_weight, bias=norm1_bias)

    tgt3, attn_weights = torch.nn.functional.multihead_attention(
        query=tgt,
        key=memory,
        value=memory,
        attn_mask=tgt_mask,
        key_padding_mask=memory_key_padding_mask,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        dropout=dropout2,
        training=False
    )

    tgt3 = tgt + tgt3
    tgt3 = torch.nn.functional.dropout(tgt3, p=dropout3, training=False)
    tgt3 = torch.nn.functional.layer_norm(tgt3, tgt3.shape[-1:], weight=norm2_weight, bias=norm2_bias)
    
    tgt4 = torch.nn.functional.linear(tgt3, linear1_weight, linear1_bias)
    tgt4 = torch.nn.functional.relu(tgt4)
    tgt4 = torch.nn.functional.dropout(tgt4, p=dropout1, training=False)
    tgt4 = torch.nn.functional.linear(tgt4, linear2_weight, linear2_bias)
    tgt4 = tgt3 + tgt4

    return tgt4, attn_weights

function_signature = {
    "name": "transformer_decoder_layer",
    "inputs": [
        ((1, 1, 4), torch.float32),
        ((1, 1, 4), torch.float32),
        ((1, 1, 1), torch.bool),
        ((1, 1), torch.bool),
        ((1, 1), torch.bool),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        ((4, 4), torch.float32),
        ((4,), torch.float32),
        (0.1, torch.float32),
        (0.1, torch.float32),
        (0.1, torch.float32),
        ('relu', torch.str),
        ((4,), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
        ((4,), torch.float32),
    ],
    "outputs": [
        ((1, 1, 4), torch.float32),
        ((1, 1, 1, 1), torch.float32),
    ]
}
