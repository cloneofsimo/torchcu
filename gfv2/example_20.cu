
#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>

#define CHECK_CUDNN(status) \
  do { \
    if (status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << std::endl; \
      exit(1); \
    } \
  } while (0)

// Function to perform replication padding in CUDA
__global__ void replication_pad3D(const float* input, float* output, int batch, int channels, int depth, int height, int width, int padding) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;
  int d = blockIdx.z * blockDim.z + threadIdx.z;

  if (b < batch && c < channels && d < depth + 2 * padding) {
    int input_d = d - padding;
    if (input_d >= 0 && input_d < depth) {
      int h = threadIdx.z;
      int w = threadIdx.w;
      if (h < height + 2 * padding && w < width + 2 * padding) {
        int input_h = h - padding;
        int input_w = w - padding;
        if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
          output[(b * channels + c) * (depth + 2 * padding) * (height + 2 * padding) * (width + 2 * padding) + d * (height + 2 * padding) * (width + 2 * padding) + h * (width + 2 * padding) + w] 
            = input[(b * channels + c) * depth * height * width + input_d * height * width + input_h * width + input_w];
        } else {
          output[(b * channels + c) * (depth + 2 * padding) * (height + 2 * padding) * (width + 2 * padding) + d * (height + 2 * padding) * (width + 2 * padding) + h * (width + 2 * padding) + w] 
            = input[(b * channels + c) * depth * height * width + input_d * height * width + (input_h >= 0 ? input_h : 0) * width + (input_w >= 0 ? input_w : 0)];
        }
      }
    } else {
      output[(b * channels + c) * (depth + 2 * padding) * (height + 2 * padding) * (width + 2 * padding) + d * (height + 2 * padding) * (width + 2 * padding) + threadIdx.z * (width + 2 * padding) + threadIdx.w] 
        = input[(b * channels + c) * depth * height * width + (input_d >= 0 ? input_d : 0) * height * width + threadIdx.z * width + threadIdx.w];
    }
  }
}

extern "C" {
void conv3d_fft_replication_pad(int num_args, ...) {
  va_list args;
  va_start(args, num_args);

  // Extract input tensor
  const float* input_tensor = va_arg(args, const float*);
  int input_tensor_dim0 = va_arg(args, int);
  int input_tensor_dim1 = va_arg(args, int);
  int input_tensor_dim2 = va_arg(args, int);
  int input_tensor_dim3 = va_arg(args, int);
  int input_tensor_dim4 = va_arg(args, int);

  // Extract weight tensor
  const float* weight_tensor = va_arg(args, const float*);
  int weight_tensor_dim0 = va_arg(args, int);
  int weight_tensor_dim1 = va_arg(args, int);
  int weight_tensor_dim2 = va_arg(args, int);
  int weight_tensor_dim3 = va_arg(args, int);
  int weight_tensor_dim4 = va_arg(args, int);

  // Extract padding value
  int padding = va_arg(args, int);

  // Extract output tensor (assuming it's preallocated)
  float* output_tensor = va_arg(args, float*);

  va_end(args);

  // CUDA context setup
  cudaDeviceSynchronize();

  // Allocate device memory
  float* d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * input_tensor_dim4 * sizeof(float));
  cudaMalloc(&d_weight, weight_tensor_dim0 * weight_tensor_dim1 * weight_tensor_dim2 * weight_tensor_dim3 * weight_tensor_dim4 * sizeof(float));
  cudaMalloc(&d_output, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * input_tensor_dim4 * sizeof(float));

  // Copy input data to device
  cudaMemcpy(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * input_tensor_dim2 * input_tensor_dim3 * input_tensor_dim4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight_tensor, weight_tensor_dim0 * weight_tensor_dim1 * weight_tensor_dim2 * weight_tensor_dim3 * weight_tensor_dim4 * sizeof(float), cudaMemcpyHostToDevice);

  // Replication Padding
  int padded_depth = input_tensor_dim2 + 2 * padding;
  int padded_height = input_tensor_dim3 + 2 * padding;
  int padded_width = input_tensor_dim4 + 2 * padding;

  int blockSize = 16;
  dim3 blockDim(blockSize, blockSize, blockSize);
  dim3 gridDim((input_tensor_dim0 * input_tensor_dim1 + blockSize - 1) / blockSize, (padded_depth + blockSize - 1) / blockSize, 1);

  replication_pad3D<<<gridDim, blockDim>>>(d_input, d_output, input_tensor_dim0, input_tensor_dim1, input_tensor_dim2, input_tensor_dim3, input_tensor_dim4, padding);

  // Convolution using cuFFT
  // 1. Create cuFFT plan
  cufftHandle plan;
  cufftPlanMany(&plan, 3, // rank of input/output arrays (3D)
                 &input_tensor_dim2, // dimensions for each rank (padded depth, padded height, padded width)
                 &input_tensor_dim2, // strides for each rank
                 &padded_depth, // distances between elements of adjacent arrays (in bytes)
                 &input_tensor_dim2 * input_tensor_dim3 * input_tensor_dim4 * sizeof(float), // distance between consecutive arrays (in bytes)
                 &padded_depth, // strides for each rank
                 &padded_depth, // distances between elements of adjacent arrays (in bytes)
                 &padded_depth * padded_height * padded_width * sizeof(float), // distance between consecutive arrays (in bytes)
                 CUFFT_C2C, // forward transform
                 input_tensor_dim0 * input_tensor_dim1, // batch size
                 CUFFT_R2C); // real to complex (for input)

  // 2. Allocate device memory for FFT output
  float* d_input_fft;
  cudaMalloc(&d_input_fft, input_tensor_dim0 * input_tensor_dim1 * padded_depth * padded_height * padded_width * sizeof(float));

  // 3. Perform FFT on input
  cufftExecR2C(plan, d_input, d_input_fft);

  // 4. Create cuFFT plan for weight (we will only do forward transform)
  cufftHandle plan_weight;
  cufftPlanMany(&plan_weight, 3, // rank of input/output arrays (3D)
                 &weight_tensor_dim2, // dimensions for each rank
                 &weight_tensor_dim2, // strides for each rank
                 &weight_tensor_dim2, // distances between elements of adjacent arrays (in bytes)
                 &weight_tensor_dim2 * weight_tensor_dim3 * weight_tensor_dim4 * sizeof(float), // distance between consecutive arrays (in bytes)
                 &weight_tensor_dim2, // strides for each rank
                 &weight_tensor_dim2, // distances between elements of adjacent arrays (in bytes)
                 &weight_tensor_dim2 * weight_tensor_dim3 * weight_tensor_dim4 * sizeof(float), // distance between consecutive arrays (in bytes)
                 CUFFT_C2C, // forward transform
                 weight_tensor_dim0 * weight_tensor_dim1, // batch size
                 CUFFT_R2C); // real to complex (for input)

  // 5. Allocate device memory for weight FFT output
  float* d_weight_fft;
  cudaMalloc(&d_weight_fft, weight_tensor_dim0 * weight_tensor_dim1 * weight_tensor_dim2 * weight_tensor_dim3 * weight_tensor_dim4 * sizeof(float));

  // 6. Perform FFT on weight
  cufftExecR2C(plan_weight, d_weight, d_weight_fft);

  // Convolution in frequency domain (element-wise multiplication)
  // Allocate a temporary buffer to store the product of the FFTs
  float* d_output_fft;
  cudaMalloc(&d_output_fft, input_tensor_dim0 * input_tensor_dim1 * padded_depth * padded_height * padded_width * sizeof(float));
  
  // Launch a kernel to perform the element-wise multiplication
  int blockSize_mul = 16;
  dim3 blockDim_mul(blockSize_mul, blockSize_mul, blockSize_mul);
  dim3 gridDim_mul((input_tensor_dim0 * input_tensor_dim1 * padded_depth * padded_height + blockSize_mul - 1) / blockSize_mul, (padded_width + blockSize_mul - 1) / blockSize_mul, 1);

  // Define the kernel for element-wise multiplication
  __global__ void mul_fft(const float* a, const float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
      c[i] = a[i] * b[i];
    }
  }

  // Launch the kernel for element-wise multiplication
  mul_fft<<<gridDim_mul, blockDim_mul>>>(d_input_fft, d_weight_fft, d_output_fft, input_tensor_dim0 * input_tensor_dim1 * padded_depth * padded_height * padded_width);

  // Perform inverse FFT on the result
  cufftExecC2R(plan, d_output_fft, d_output);

  // Free temporary buffer
  cudaFree(d_output_fft);

  // Crop output tensor to original size
  int output_depth = input_tensor_dim2;
  int output_height = input_tensor_dim3;
  int output_width = input_tensor_dim4;

  int blockSize_crop = 16;
  dim3 blockDim_crop(blockSize_crop, blockSize_crop, blockSize_crop);
  dim3 gridDim_crop((input_tensor_dim0 * input_tensor_dim1 + blockSize_crop - 1) / blockSize_crop, (output_depth + blockSize_crop - 1) / blockSize_crop, 1);

  // Define the kernel for cropping
  __global__ void crop3D(const float* input, float* output, int batch, int channels, int depth, int height, int width, int padding) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (b < batch && c < channels && d < depth) {
      int input_d = d + padding;
      int h = threadIdx.z;
      int w = threadIdx.w;
      output[(b * channels + c) * depth * height * width + d * height * width + h * width + w] = input[(b * channels + c) * padded_depth * padded_height * padded_width + input_d * padded_height * padded_width + (h + padding) * padded_width + (w + padding)];
    }
  }

  // Launch the kernel for cropping
  crop3D<<<gridDim_crop, blockDim_crop>>>(d_output, d_input, input_tensor_dim0, input_tensor_dim1, output_depth, output_height, output_width, padding);

  // ReLU activation
  // We can use a simple kernel for ReLU
  __global__ void relu(float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
      output[i] = (output[i] > 0.0f) ? output[i] : 0.0f;
    }
  }

  int blockSize_relu = 16;
  dim3 blockDim_relu(blockSize_relu, 1, 1);
  dim3 gridDim_relu((input_tensor_dim0 * input_tensor_dim1 * output_depth * output_height * output_width + blockSize_relu - 1) / blockSize_relu, 1, 1);

  relu<<<gridDim_relu, blockDim_relu>>>(d_input, input_tensor_dim0 * input_tensor_dim1 * output_depth * output_height * output_width);

  // Copy result back to host
  cudaMemcpy(output_tensor, d_input, input_tensor_dim0 * input_tensor_dim1 * output_depth * output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  cudaFree(d_input_fft);
  cudaFree(d_weight_fft);

  // Destroy cuFFT plans
  cufftDestroy(plan);
  cufftDestroy(plan_weight);
}
} // extern "C"
