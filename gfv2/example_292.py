
import torch

def attention_prelu_bucketize_fp16(input_tensor: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, weights: torch.Tensor, bucket_size: int) -> torch.Tensor:
    """
    Performs attention, PReLU activation, and bucketization on input tensor.
    """
    input_fp16 = input_tensor.to(torch.float16)
    query_fp16 = query.to(torch.float16)
    key_fp16 = key.to(torch.float16)
    value_fp16 = value.to(torch.float16)
    weights_fp16 = weights.to(torch.float16)

    # Attention
    attention_scores = torch.matmul(query_fp16, key_fp16.transpose(1, 2))
    attention_weights = torch.softmax(attention_scores, dim=-1)
    context = torch.matmul(attention_weights, value_fp16)

    # PReLU
    output = torch.nn.functional.prelu(context, weights_fp16)

    # Bucketize
    output = torch.bucketize(output, torch.arange(0, 10, bucket_size).to(torch.float16))

    return output.to(torch.float32)

function_signature = {
    "name": "attention_prelu_bucketize_fp16",
    "inputs": [
        ((10, 5), torch.float32),
        ((10, 3), torch.float32),
        ((10, 3), torch.float32),
        ((10, 5), torch.float32),
        ((3,), torch.float32),
        (10, )
    ],
    "outputs": [
        ((10, 5), torch.float32),
    ]
}
