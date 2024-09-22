
import torch
import torchaudio

def torch_audio_resynthesis_bfloat16_function(input_tensor: torch.Tensor, 
                                            trace_tensor: torch.Tensor) -> torch.Tensor:
    """
    Resynthesizes audio using a trace tensor in bfloat16 precision.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    trace_bf16 = trace_tensor.to(torch.bfloat16)

    # Resynthesis using torchaudio
    resynthesized_audio_bf16 = torchaudio.functional.resynthesis(input_bf16, trace_bf16)

    return resynthesized_audio_bf16.to(torch.float32)

function_signature = {
    "name": "torch_audio_resynthesis_bfloat16_function",
    "inputs": [
        ((1, 16000), torch.float32),
        ((1, 16000), torch.float32),
    ],
    "outputs": [
        ((1, 16000), torch.float32)
    ]
}
