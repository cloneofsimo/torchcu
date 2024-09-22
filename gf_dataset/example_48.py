
import torch

def torch_audio_processing(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
                           filter_lengths: list, filter_deltas: list) -> torch.Tensor:
    """
    Processes audio input using a series of log-filtering, adaptive max pooling, and convolutional operations.
    """
    # Convert to bfloat16 for efficiency
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16)

    # Log-filtering
    log_filter_outputs = []
    for filter_length, filter_delta in zip(filter_lengths, filter_deltas):
        log_filter_output = torch.log(1 + torch.abs(input_bf16))
        log_filter_output = torch.nn.functional.adaptive_max_pool1d(log_filter_output, filter_length)
        log_filter_outputs.append(log_filter_output)

    # Combine log-filter outputs
    combined_log_filter = torch.cat(log_filter_outputs, dim=1)

    # Convolutional layer
    conv_output = torch.nn.functional.conv1d(combined_log_filter, weight_bf16, bias=bias_bf16)
    conv_output = torch.nn.functional.relu(conv_output)

    # Add-CMUL
    output = torch.addcmul(conv_output, 1, conv_output, 0.5)

    # Convert back to float32
    return output.to(torch.float32)


function_signature = {
    "name": "torch_audio_processing",
    "inputs": [
        ((1, 20, 16000), torch.float32),
        ((10, 20, 1), torch.float32),
        ((10,), torch.float32),
        ((2,), torch.int32),
        ((2,), torch.int32)
    ],
    "outputs": [
        ((1, 10, 8000), torch.float32),
    ]
}
