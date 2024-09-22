
import shutil
from together import Together
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
from judge import judge_it

load_dotenv()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--force-regenerate", action="store_true")
args = parser.parse_args()

force_regenerate = args.force_regenerate

__CURR_DIR__ = Path(__file__).parent

client = Together(api_key=os.environ['TOGETHER_API_KEY'])

arena_data_dir = __CURR_DIR__ / "test_py_files"

TEMP_ARENA_DATA_DIR = __CURR_DIR__ / "generated_arena_data"


def split_cu_code(text: str) -> str:
    """Split the cu code into function definitions and other code."""
    # find codeblock ```c++
    for extension in ["```c++", "```cpp", "```cuda", "```cu", "```c"]:
        if extension in text:
            break

    try:
        cu_code = text.split(extension)[1].split("```")[0]
    except IndexError:
        breakpoint()
        raise ValueError("No code block found")
    return cu_code


@dataclass
class LLM:
    model: str
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1
    stop: tuple[str] = ("<|eot_id|>", "<|eom_id|>")
    _chat_template: str = """
    Given a following python pytorch code, generate the cuda kernel that is equivalent.
    Here is an example python code:
    ```python
    import torch

    def linear_bfloat16_function(input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        \"""
        Perform a simple linear transformation (matrix multiplication) and activation using bfloat16.
        \"""
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

    And your output should be like this:

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
    """.strip()

    source_code_template: str = """
    Return the output, func.cu in codeblocks with the exact structure below. The name of the function should be the same as the function name in the signature.
    You don't have to explain after the codeblocks. Dont explain before and after the codeblocks. Just return the code.
    ```python
    {source_code}
    ```
    """.strip()

    def generate(self, source_file: Path):
        with open(source_file, "r") as f:
            source_code = f.read()

        source_code_prompt = self.source_code_template.format(source_code=source_code)
        prompt = self._chat_template + source_code_prompt

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
        )

        return response.choices[0].message.content.strip()



MODELS = [
    "codellama/CodeLlama-34b-Instruct-hf",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
][-1:]



if __name__ == "__main__":
    source_files = list(arena_data_dir.glob("*.py"))
    source_codes = {
        file.stem: file.read_text() for file in source_files
    }
    
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="Arena")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Score", style="green", no_wrap=True)

    TEMP_ARENA_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        model_dir = TEMP_ARENA_DATA_DIR / model.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        llm = LLM(model=model)
        model_score = 0.0
        for file in source_files:
            py_file = model_dir / file.name

            shutil.copy(file, model_dir)

            output_file = model_dir / (file.stem + ".cu")
            if not output_file.exists() or force_regenerate:
                for _ in range(3):
                    output = llm.generate(py_file)
                    try:
                        output_file.write_text(split_cu_code(output))
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            model_score += judge_it(py_file)

        model_score /= len(source_files)

        table.add_row(model, f"{model_score:.2f}")

    console.print(table)
    
