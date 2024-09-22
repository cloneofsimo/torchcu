
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdarg.h>

#define  FLOAT2HALF(x)   ((half)((x) * 65536.0f))
#define  HALF2FLOAT(x)   ((float)((x) / 65536.0f))

// Helper function to convert float to half
__device__ __forceinline__ half float_to_half(float f) {
    return FLOAT2HALF(f);
}

// Helper function to convert half to float
__device__ __forceinline__ float half_to_float(half h) {
    return HALF2FLOAT(h);
}

__global__ void attention_kernel(const half* input, const half* target, float* output_loss, half* output_alpha_max,
                                int batch_size, int seq_len, int hidden_size, int attention_size, float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        half sum_loss = 0.0f;
        half max_alpha = -FLT_MAX;
        
        for (int j = 0; j < seq_len; ++j) {
            half context[128] = {0.0f};
            half alpha[32] = {0.0f};
            
            // Calculate attention context
            for (int k = 0; k < hidden_size; ++k) {
                context[k] = input[i * seq_len * hidden_size + j * hidden_size + k];
            }
            
            // Apply log_softmax with temperature
            half log_softmax[128] = {0.0f};
            for (int k = 0; k < hidden_size; ++k) {
                log_softmax[k] = exp(context[k] / temperature);
            }
            float sum = 0.0f;
            for (int k = 0; k < hidden_size; ++k) {
                sum += log_softmax[k];
            }
            for (int k = 0; k < hidden_size; ++k) {
                log_softmax[k] = log(log_softmax[k] / sum);
            }
            
            // Calculate margin ranking loss
            sum_loss +=  max(0.0f, 0.2f + log_softmax[i] - log_softmax[j]);
            
            // Apply max filter to attention weights
            alpha[j] = log_softmax[i];
            if (alpha[j] > max_alpha) {
                max_alpha = alpha[j];
            }
        }
        output_loss[i] = half_to_float(sum_loss / seq_len);
        output_alpha_max[i] = max_alpha;
    }
}

extern "C" {

void my_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);
    
    // Extract input tensors
    const half* input_tensor = va_arg(args, const half*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);
    int input_tensor_dim2 = va_arg(args, int);
    
    const half* target_tensor = va_arg(args, const half*);
    int target_tensor_dim0 = va_arg(args, int);
    int target_tensor_dim1 = va_arg(args, int);
    int target_tensor_dim2 = va_arg(args, int);
    
    // Extract output tensors (assuming they are preallocated)
    float* output_loss = va_arg(args, float*);
    half* output_alpha_max = va_arg(args, half*);
    
    va_end(args);
    
    int batch_size = input_tensor_dim0;
    int seq_len = input_tensor_dim1;
    int hidden_size = input_tensor_dim2;
    int attention_size = 64;
    float temperature = 2.0f;
    
    // Launch kernel
    dim3 threadsPerBlock(128);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    attention_kernel<<<numBlocks, threadsPerBlock>>>(input_tensor, target_tensor, output_loss, output_alpha_max,
                                                batch_size, seq_len, hidden_size, attention_size, temperature);
    
    cudaDeviceSynchronize();
}

}  // extern "C"
