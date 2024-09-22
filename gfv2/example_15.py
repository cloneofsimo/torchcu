
import torch
import torch.nn.functional as F

def torch_transformer_decoder_fp16_function(input_tensor: torch.Tensor, 
                                           memory_tensor: torch.Tensor,
                                           query_tensor: torch.Tensor, 
                                           key_tensor: torch.Tensor, 
                                           value_tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs a Transformer decoder step with FP16 precision.
    """
    input_fp16 = input_tensor.to(torch.float16)
    memory_fp16 = memory_tensor.to(torch.float16)
    query_fp16 = query_tensor.to(torch.float16)
    key_fp16 = key_tensor.to(torch.float16)
    value_fp16 = value_tensor.to(torch.float16)

    # Multi-head attention with memory
    attention_output = F.multihead_attention(
        query_fp16, memory_fp16, memory_fp16, 
        attn_mask=None,
        key_padding_mask=None,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        dropout=0.0,
        training=False,
        need_weights=False
    )

    # Combine input and attention output
    decoder_input = input_fp16 + attention_output[0]

    # Feedforward network
    decoder_output = F.linear(decoder_input, weight=torch.ones(1, 1, dtype=torch.float16).cuda())

    return decoder_output.to(torch.float32)

function_signature = {
    "name": "torch_transformer_decoder_fp16_function",
    "inputs": [
        ((1, 5, 10), torch.float32),
        ((1, 10, 10), torch.float32),
        ((1, 5, 10), torch.float32),
        ((1, 10, 10), torch.float32),
        ((1, 10, 10), torch.float32)
    ],
    "outputs": [
        ((1, 5, 1), torch.float32),
    ]
}
