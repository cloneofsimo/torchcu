import logging
import os
import random

from generator.example_models import models
from generator.llm import LLM

prompt_write_functions = """Here are example pytorch codes.
{examples}

Now, let's write some pytorch functions.

Constraints:
1. Takes torch tensors and returns a torch tensor or tuple of them.
It should be a function that takes torch tensors as parameters and,
returns a torch tensor or tuple of torch tensors as output.

2. The function should be self-contained.
Make it self-contained and independent of other functions.
Every functions or classes should be defined in the code.
Do not import any external functions or classes, except torch and numpy.

3. Answer has a specific format.
Each function in the answer should be formatted like below:
### <summary of the function>
```python
import torch

def <function_name>(<parameter_name>: torch.Tensor, <parameter_name>: torch.Tensor, ...) -> torch.Tensor:
    # Your code here
```

4. Make it not too simple.
The function should not be too simple like adding two tensors.
It should have some complexity like using multiple operations or functions.
Refer to the examples at the top.

Make sure the function takes and returns torch tensors and is self-contained.
Also, your example should have following concepts:
{concepts}

Make 3 functions.
"""

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
    "unstructured_sparsity",
]


important_concepts = [
    "fp32",
    "bf16",
    "fp16",
    "int8",
    "backward",
    "forward",
    "inplace",
    "cutlass",
    "cudnn",
    "pure cu",
]

logger = logging.getLogger()


def get_concepts():
    num_choices = random.randint(1, 4)
    concepts_to_use = random.sample(concepts, num_choices) + random.sample(
        important_concepts, num_choices
    )
    return ", ".join(concepts_to_use)


def generate_functions(
    reference_model: str,
    output_md_dir: str,
    output_py_dir: str,
    max_reference_files: int = 3,
    reuse_existing_md: bool = False,
) -> list[str]:
    md = ""
    md_path = f"{output_md_dir}/learn_from_{reference_model}.md"
    if reuse_existing_md and os.path.exists(md_path):
        md = open(md_path, "r").read()
    else:
        llm = LLM()

        examples = ""
        modelfiles = models[reference_model]
        count = 0
        for filename, content in modelfiles.items():
            if count >= max_reference_files:
                break
            examples += f"filename: {filename}\n{content}\n\n"
            count += 1

        messages = [
            {
                "role": "user",
                "content": prompt_write_functions.format(
                    examples=examples, concepts=get_concepts()
                ),
            }
        ]

        logger.info(
            f"Generating functions based on {reference_model} ({len(modelfiles)} files)"
        )
        with open(md_path, "w") as f:
            for response in llm.generate(messages):
                md += response
                f.write(response)

        logger.info(f"Generated markdown file: {md_path} (token usage: {llm.usage})")

    filepaths = []
    snippets = [s for s in md.split("###") if s != ""]
    logger.info(f"Generated {len(snippets)} snippets")
    for snippet in snippets:
        try:
            name = snippet.split("\n")[0].strip().replace(" ", "_").lower()
            code = snippet.split("```python")[1].split("```")[0].strip()
        except IndexError:
            logger.warning(f"Failed to parse snippet, model output: {snippet}")
            continue

        filepath = f"{output_py_dir}/{reference_model}_{name}.py"
        if os.path.exists(filepath):
            filepath = (
                f"{output_py_dir}/{reference_model}_{name}_{random.randint(0, 1000)}.py"
            )
        with open(filepath, "w") as f:
            f.write(code)
        filepaths.append(filepath)

        logger.info(f"Saved snippet to {filepath}")

    return filepaths
