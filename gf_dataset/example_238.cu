
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define CHECK(x) do {                                      \
    if ((x) != cudaSuccess) {                             \
        std::cerr << "Error: " << cudaGetErrorString(x) << std::endl; \
        exit(1);                                          \
    }                                                     \
} while (0)

extern "C" {

void torch_function(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    // Extract arguments
    const float* input_tensor = va_arg(args, const float*);
    int input_tensor_dim0 = va_arg(args, int);
    int input_tensor_dim1 = va_arg(args, int);

    int window_size = va_arg(args, int);
    int hop_size = va_arg(args, int);
    int n_fft = va_arg(args, int);
    int mel_bins = va_arg(args, int);
    int sample_rate = va_arg(args, int);

    const float* mean = va_arg(args, const float*);
    int mean_dim0 = va_arg(args, int);
    int mean_dim1 = va_arg(args, int);

    const float* std = va_arg(args, const float*);
    int std_dim0 = va_arg(args, int);
    int std_dim1 = va_arg(args, int);

    // Extract output tensor (assuming preallocated)
    float* output = va_arg(args, float*);

    va_end(args);

    // CUDA context and handle
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudnnHandle_t cudnnHandle;
    CHECK(cudnnCreate(&cudnnHandle));

    // Input and output descriptions
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK(cudnnCreateTensorDescriptor(&outputDesc));

    CHECK(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 1,
                                    (int64_t[]){input_tensor_dim0, input_tensor_dim1}, 
                                    (int64_t[]){1, 1}));
    CHECK(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT, 3,
                                    (int64_t[]){input_tensor_dim0, mel_bins, n_fft / 2 + 1}, 
                                    (int64_t[]){1, 1, 1}));

    // Allocate device memory
    float* d_input, *d_mean, *d_std, *d_output;
    CHECK(cudaMalloc(&d_input, input_tensor_dim0 * input_tensor_dim1 * sizeof(float)));
    CHECK(cudaMalloc(&d_mean, mean_dim0 * mean_dim1 * sizeof(float)));
    CHECK(cudaMalloc(&d_std, std_dim0 * std_dim1 * sizeof(float)));
    CHECK(cudaMalloc(&d_output, input_tensor_dim0 * mel_bins * (n_fft / 2 + 1) * sizeof(float)));

    // Copy input data to device
    CHECK(cudaMemcpyAsync(d_input, input_tensor, input_tensor_dim0 * input_tensor_dim1 * sizeof(float), 
                    cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_mean, mean, mean_dim0 * mean_dim1 * sizeof(float), 
                    cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(d_std, std, std_dim0 * std_dim1 * sizeof(float), 
                    cudaMemcpyHostToDevice, stream));

    // STFT using cuDNN
    cudnnTensorDescriptor_t stftDesc;
    CHECK(cudnnCreateTensorDescriptor(&stftDesc));
    CHECK(cudnnSetTensorNdDescriptor(stftDesc, CUDNN_DATA_COMPLEX_FLOAT, 3,
                                    (int64_t[]){input_tensor_dim0, n_fft / 2 + 1, 129}, 
                                    (int64_t[]){1, 1, 1}));

    cudnnSTFTDescriptor_t stftPlan;
    CHECK(cudnnCreateSTFTDescriptor(&stftPlan));
    CHECK(cudnnSetSTFTDescriptor(stftPlan, window_size, hop_size, n_fft, 
                                     CUDNN_STFT_MODE_CUFFT_FORWARD, 
                                     CUDNN_STFT_INPUT_FORMAT_COMPLEX_FLOAT,
                                     CUDNN_STFT_OUTPUT_FORMAT_COMPLEX_FLOAT,
                                     CUDNN_STFT_PAD_MODE_ZEROS));

    // Allocate memory for the complex STFT output
    float* d_stft_output;
    CHECK(cudaMalloc(&d_stft_output, input_tensor_dim0 * (n_fft / 2 + 1) * 129 * sizeof(float) * 2));

    // Perform STFT on the device
    CHECK(cudnnSTFTForward(cudnnHandle, stftPlan, 
                            inputDesc, d_input, 
                            stftDesc, d_stft_output));

    // Magnitude calculation
    CHECK(cudnnTransformTensor(cudnnHandle, CUDNN_OP_MAG_SQUARE, 
                              stftDesc, d_stft_output, 
                              outputDesc, d_output));

    // Mel-spectrogram calculation
    cudnnTensorDescriptor_t melDesc;
    CHECK(cudnnCreateTensorDescriptor(&melDesc));
    CHECK(cudnnSetTensorNdDescriptor(melDesc, CUDNN_DATA_FLOAT, 3,
                                    (int64_t[]){input_tensor_dim0, mel_bins, n_fft / 2 + 1}, 
                                    (int64_t[]){1, 1, 1}));

    cudnnFilterDescriptor_t melFilterDesc;
    CHECK(cudnnCreateFilterDescriptor(&melFilterDesc));
    // Assuming you have precomputed mel filter banks on the host
    // (The actual filter bank creation is not shown here)
    float* h_melFilter;
    CHECK(cudaMalloc(&h_melFilter, mel_bins * (n_fft / 2 + 1) * sizeof(float)));
    // ... (Load the mel filter bank to h_melFilter)

    float* d_melFilter;
    CHECK(cudaMalloc(&d_melFilter, mel_bins * (n_fft / 2 + 1) * sizeof(float)));
    CHECK(cudaMemcpyAsync(d_melFilter, h_melFilter, mel_bins * (n_fft / 2 + 1) * sizeof(float), 
                    cudaMemcpyHostToDevice, stream));

    CHECK(cudnnSetFilterNdDescriptor(melFilterDesc, CUDNN_DATA_FLOAT, 1,
                                    (int64_t[]){mel_bins, (n_fft / 2 + 1)}, 
                                    (int64_t[]){1, 1}));

    // Calculate the mel-spectrogram
    CHECK(cudnnConvolutionForward(cudnnHandle,
                                 CUDNN_CONVOLUTION_MODE_CROSS_CORRELATION,
                                 CUDNN_DATA_FLOAT, inputDesc, d_output,
                                 CUDNN_DATA_FLOAT, melFilterDesc, d_melFilter,
                                 CUDNN_DATA_FLOAT, outputDesc, d_output,
                                 nullptr)); 

    // Logarithmic scaling
    CHECK(cudnnTransformTensor(cudnnHandle, CUDNN_OP_LOG, outputDesc, d_output, outputDesc, d_output));

    // Normalize with mean and std
    // Assuming mean and std are already on the device (d_mean, d_std)
    CHECK(cudnnTransformTensor(cudnnHandle, CUDNN_OP_TENSOR_ADD, outputDesc, d_mean, outputDesc, d_output));
    CHECK(cudnnTransformTensor(cudnnHandle, CUDNN_OP_TENSOR_DIV, outputDesc, d_std, outputDesc, d_output));

    // Copy the result back to host
    CHECK(cudaMemcpyAsync(output, d_output, input_tensor_dim0 * mel_bins * (n_fft / 2 + 1) * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream));

    // Synchronize the stream
    CHECK(cudaStreamSynchronize(stream));

    // Free device memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_mean));
    CHECK(cudaFree(d_std));
    CHECK(cudaFree(d_output));
    CHECK(cudaFree(d_stft_output));
    CHECK(cudaFree(d_melFilter));

    // Destroy descriptors and handles
    CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK(cudnnDestroyTensorDescriptor(stftDesc));
    CHECK(cudnnDestroyTensorDescriptor(melDesc));
    CHECK(cudnnDestroyFilterDescriptor(melFilterDesc));
    CHECK(cudnnDestroySTFTDescriptor(stftPlan));
    CHECK(cudnnDestroy(cudnnHandle));
    CHECK(cudaStreamDestroy(stream));
}

}  // extern "C"
