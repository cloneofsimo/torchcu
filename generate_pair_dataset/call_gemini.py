import os
import google.generativeai as genai
import typer

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)


prompt = '''
Make a example torch function and its signiture and its cuda transfiled code.

For example. here is the example torch code and its transpile cuda code. Notice how for cuda code, we have argument in order of (data1, shape10, shape11, shape12, data2, ....)

```python
import torch

def linear_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Perform a simple linear transformation (matrix multiplication) and activation using bfloat16.
    """
    input_bf16 = input_tensor.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    output = torch.matmul(input_bf16, weight_bf16.t())
    return torch.relu(output).to(torch.float32)

function_signature = {
    "name": "linear_bfloat16_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32)
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}
```

Transformed into:

```c++
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>  // Add this for va_list, va_start, va_end

// Helper function to convert float to __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bfloat16(float f) {
    return __float2bfloat16(f);
}

// Helper function to convert __nv_bfloat16 to float
__device__ __forceinline__ float bfloat16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

// CUDA kernel for matrix multiplication and ReLU using bfloat16
__global__ void matmul_relu_kernel_bf16(const float* input_tensor, const float* weight, float* output, 
                                        int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            __nv_bfloat16 a = float_to_bfloat16(input_tensor[row * k + i]);
            __nv_bfloat16 b = float_to_bfloat16(weight[col * k + i]);  // Transposed access
            sum += bfloat16_to_float(__hmul(a, b));
        }
        output[row * n + col] = fmaxf(sum, 0.0f);  // ReLU activation
    }
}

extern "C" {

void linear_bfloat16_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract input tensor
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    // Extract weight tensor
    const float* weight = va_arg(args, const float*);
    int weight_dim0 = va_arg(args, int);
    int weight_dim1 = va_arg(args, int);

    // Extract output tensor (assuming it's preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    int batch_size = input_tensor_dim0;
    int input_dim = input_tensor_dim1;
    int output_dim = weight_dim0;

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, output_dim * input_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input_tensor, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_relu_kernel_bf16<<<numBlocks, threadsPerBlock>>>(
        d_input, d_weight, d_output, batch_size, output_dim, input_dim
    );

    // Copy result back to host
    cudaMemcpy(output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

}  // extern "C"
```


Make one more example like this. Return the output, func.py and func.cu in codeblocks. You don't have to explain after the codeblocks.
constraint is that your torch function should always have tensors as input, and (optionally) list of tensor as output.
'''

def make_exmample(use_these_concepts=["flash attention", "sort", "linear"]):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt_with_concepts = (
        prompt
        + "\n\n Also, your example should have following concepts: "
        + ", ".join(use_these_concepts)
        + "Return Only One Example. For cuda code, cutlass or cudnn if possible to optimize further. DO NOT SKIP SINGLE LINE. IMPLELEMT EVERYTHING."
    )
    response = model.generate_content(prompt_with_concepts)
    return response.text


import random


def make_set_of_examples(num_examples: int = 4000, output_dir: str = "example_dataset"):
    concepts = [
        "attention",
        "double linear",
        "sort",
        "max",
        "min",
        "sum",
        "mean",
        "abs",
        "exp",
        "log",
        "sigmoid",
        "tanh",
        "relu",
        "softmax",
        "log_softmax",
        "dropout",
        "batch_norm",
        "layer_norm",
        "group_norm",
        "instance_norm",
        "conv1d",
        "conv2d",
        "conv3d",
        "transposed_conv1d",
        "transposed_conv2d",
        "transposed_conv3d",
        "avg_pool1d",
        "avg_pool2d",
        "avg_pool3d",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "adaptive_avg_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "linear",
        "bilinear",
        "einsum",
        "matmul",
        "mm",
        "dot",
        "bmm",
        "add",
        "subtract",
        "multiply",
        "divide",
        "true_divide",
        "floor_divide",
        "pow",
        "sqrt",
        "square",
        "round",
        "floor",
        "ceil",
        "clamp",
        "clip",
        "lerp",
        "mean",
        "var",
        "std",
        "all",
        "any",
        "eq",
        "ne",
        "ge",
        "gt",
        "le",
        "lt",
        "transpose",
        "permute",
        "expand",
        "repeat",
        "flatten",
        "view",
        "reshape",
        "squeeze",
        "unsqueeze",
        "index_select",
        "masked_select",
        "gather",
        "scatter",
        "scatter_add",
        "pad",
        "constant_pad",
        "reflection_pad",
        "replication_pad",
        "cross",
        "det",
        "inverse",
        "svd",
        "cholesky",
        "qr",
        "eig",
        "matrix_exp",
        "meshgrid",
        "broadcast_tensors",
        "broadcast_to",
        "roll",
        "unfold",
        "fold",
        "conv_tbc",
        "pixel_shuffle",
        "pixel_unshuffle",
        "interpolate",
        "upsample",
        "grid_sample",
        "affine_grid",
        "manual_seed",
        "normal",
        "uniform",
        "bernoulli",
        "exponential",
        "poisson",
        "multinomial",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
        "cross_entropy",
        "mse_loss",
        "l1_loss",
        "smooth_l1_loss",
        "kl_div",
        "nll_loss",
        "ctc_loss",
        "hinge_embedding_loss",
        "margin_ranking_loss",
        "soft_margin_loss",
        "cosine_embedding_loss",
        "multi_margin_loss",
        "multi_label_margin_loss",
        "pairwise_distance",
        "triplet_margin_loss",
        "affine_grid_generator",
        "grid_sampler",
        "relu6",
        "hardshrink",
        "hardtanh",
        "prelu",
        "rrelu",
        "selu",
        "celu",
        "gelu",
        "softplus",
        "softshrink",
        "softsign",
        "threshold",
        "logsigmoid",
        "logit",
        "elu",
        "sigmoid_focal_loss",
        "hardsigmoid",
        "mish",
        "swish",
        "adaptive_log_softmax",
        "cosine_similarity",
        "pairwise_euclidean_distance",
        "pairwise_manhattan_distance",
        "pairwise_chebyshev_distance",
        "pairwise_hamming_distance",
        "tensor_slice",
        "nonzero",
        "where",
        "bucketize",
        "isclose",
        "isin",
        "unique",
        "topk",
        "kthvalue",
        "mode",
        "median",
        "norm",
        "frobenius_norm",
        "matrix_rank",
        "trace",
        "cumsum",
        "cumprod",
        "logsumexp",
        "softmin",
        "eye",
        "arange",
        "linspace",
        "logspace",
        "ones",
        "zeros",
        "empty",
        "full",
        "tensor",
        "as_tensor",
        "from_numpy",
        "from_file",
        "ones_like",
        "zeros_like",
        "full_like",
        "empty_like",
        "identity",
        "diag",
        "diagflat",
        "block_diag",
        "kronecker_product",
        "outer_product",
        "inner_product",
        "dot_product",
        "hadamard_product",
        "elementwise_min",
        "elementwise_max",
        "elementwise_sum",
        "elementwise_diff",
        "elementwise_product",
        "elementwise_div",
        "elementwise_pow",
        "addcmul",
        "addcdiv",
        "addmv",
        "addr",
        "baddbmm",
        "bmm_out",
        "chain_matmul",
        "einsum_broadcast",
        "einsum_contraction",
        "einsum_summation",
        "einsum_transpose",
        "einsum_outer",
        "einsum_inner",
        "fft",
        "ifft",
        "rfft",
        "irfft",
        "fftshift",
        "ifftshift",
        "dft",
        "idft",
        "conv_fft",
        "conv_ifft",
        "fft_conv1d",
        "fft_conv2d",
        "fft_conv3d",
        "sobel_filter",
        "laplace_filter",
        "gaussian_filter",
        "median_filter",
        "bilateral_filter",
        "box_filter",
        "disk_filter",
        "log_filter",
        "max_filter",
        "min_filter",
        "mode_filter",
        "gradient_magnitude",
        "hessian_matrix",
        "image_gradient",
        "image_jacobian",
        "image_laplacian",
        "sobel_gradient",
        "roberts_cross_gradient",
        "prewitt_gradient",
        "scharr_gradient",
        "canny_edge_detection",
        "non_maximum_suppression",
        "morphological_dilation",
        "morphological_erosion",
        "morphological_opening",
        "morphological_closing",
        "distance_transform",
        "watershed_segmentation",
        "fourier_transform",
        "inverse_fourier_transform",
        "wavelet_transform",
        "discrete_wavelet_transform",
        "inverse_discrete_wavelet_transform",
        "continuous_wavelet_transform",
        "hilbert_transform",
        "stft",
        "istft",
        "waveform_analysis",
        "spectrogram",
        "mel_spectrogram",
        "mfcc",
        "audio_normalization",
        "audio_downsampling",
        "audio_upsampling",
        "audio_compression",
        "audio_decompression",
        "audio_denoising",
        "pitch_shift",
        "time_stretch",
        "pitch_correction",
        "harmonic_percussive_separation",
        "vocoding",
        "audio_resynthesis",
        "cross_fade",
        "fading_in",
        "fading_out",
        "audio_clipping",
        "noise_injection",
        "signal_shift",
        "signal_envelope",
        "zero_crossing_rate",
        "energy_computation",
        "root_mean_square_energy",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_contrast",
        "flash attention",
        "multi_head_attention",
        "self_attention",
        "scaled_dot_product_attention",
        "multi_scale_attention",
        "linear_attention",
        "window_attention",
        "cross_attention",
        "causal_attention",
        "local_attention",
        "global_attention",
        "vision_transformer",
        "swin_transformer",
        "detr_transformer",
        "transformer_encoder",
        "transformer_decoder",
        "transformer_layer",
        "masked_attention",
        "attention_mask",
        "gradient_checkpointing",
        "contrastive_loss",
        "supervised_contrastive_loss",
        "simclr_loss",
        "triplet_loss",
        "center_loss",
        "arcface_loss",
        "cosface_loss",
        "adaptive_avg_pool",
        "adaptive_max_pool",
        "depthwise_conv2d",
        "separable_conv2d",
        "conv1d_fft",
        "conv2d_fft",
        "conv3d_fft",
        "lightweight_conv",
        "dynamic_conv",
        "deformable_conv",
        "grouped_conv",
        "depthwise_separable_conv",
        "coord_conv",
        "gated_linear_units",
        "gumbel_softmax",
        "dynamic_positional_encoding",
        "relative_positional_encoding",
        "rotary_positional_encoding",
        "learned_positional_encoding",
        "geglu",
        "swiglu",
        "pre_activation",
        "activation_fn",
        "se_module",
        "spatial_attention",
        "channel_attention",
        "coord_attention",
        "mask_attention",
        "layer_scaling",
        "layer_scaling_decay",
        "weight_standardization",
        "gradient_clipping",
        "fused_layer_norm",
        "fused_dropout",
        "fused_softmax",
        "fused_relu",
        "fused_gelu",
        "fused_linear",
        "token_mixing",
        "feature_mixing",
        "attention_heads",
        "attention_weights",
        "softmax_temperature",
        "log_softmax_temperature",
        "gradient_accumulation",
        "gradient_sparsity",
        "gradient_quantization",
        "gradient_precision_scaling",
        "memory_efficiency",
        "inplace_operations",
        "checkpointing_operations",
        "accelerated_operations",
        "auto_mixed_precision",
        "quantization_aware_training",
        "post_training_quantization",
        "knowledge_distillation",
        "teacher_student_training",
        "self_supervised_learning",
        "contrastive_learning",
        "adversarial_training",
        "robust_loss",
        "regularization",
        "ridge_regularization",
        "lasso_regularization",
        "drop_path",
        "stochastic_depth",
        "cutout",
        "mixup",
        "cutmix",
        "label_smoothing",
        "gradient_penalty",
        "wasserstein_loss",
        "orthogonal_regularization",
        "low_rank_approximation",
        "tensor_decomposition",
        "hyperparameter_optimization",
        "model_pruning",
        "sparse_training",
        "pruning_mask",
        "weight_sparsity",
        "structured_sparsity",
        "unstructured_sparsity"
    ]
    
    important_concepts = ['fp32', 'bf16', 'fp16', 'int8', 'backward', 'forward', 'inplace', 'cutlass', 'cudnn', 'pure cu']

    # append on the existing ones
    os.makedirs(output_dir, exist_ok=True)
    existing_data = os.listdir(output_dir)
    offset = len(existing_data)
    for i in range(num_examples):
        
        num_choices = random.randint(1, 4)
        concepts_to_use = random.sample(concepts, num_choices) + random.sample(important_concepts, num_choices)
        
        output = make_exmample(use_these_concepts=concepts_to_use)
        with open(f"{output_dir}/example_{i + offset}.txt", "w") as f:
            print(output, i + offset)
            f.write(output)


if __name__ == "__main__":
    typer.run(make_set_of_examples)
