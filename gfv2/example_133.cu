
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <stdarg.h>

__global__ void harmonic_percussive_separation_int8_kernel(const int8_t* audio, int8_t* harmonic, int8_t* percussive, int n, int t, int iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t) {
        int8_t* h = harmonic + i;
        int8_t* p = percussive + i;
        int8_t* a = audio + i;
        *h = 0;
        *p = 0;

        for (int iter = 0; iter < iterations; ++iter) {
            int8_t cross_correlation = 0;
            for (int j = 0; j < t; ++j) {
                cross_correlation += *h * *(p + (t - j - 1));
            }

            *h = (int8_t) ((float)(*h + *a + cross_correlation) * 0.5f);
            *p = (int8_t) ((float)(*p + *a - cross_correlation) * 0.5f);
        }
    }
}

extern "C" {

void harmonic_percussive_separation_int8(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float* audio = va_arg(args, const float*);
    int n = va_arg(args, int);
    int t = va_arg(args, int);
    int iterations = va_arg(args, int);

    float* harmonic = va_arg(args, float*);

    va_end(args);

    int8_t* d_audio;
    int8_t* d_harmonic;
    int8_t* d_percussive;

    cudaMalloc(&d_audio, t * sizeof(int8_t));
    cudaMalloc(&d_harmonic, t * sizeof(int8_t));
    cudaMalloc(&d_percussive, t * sizeof(int8_t));

    cudaMemcpy(d_audio, audio, t * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(128);
    dim3 numBlocks((t + threadsPerBlock.x - 1) / threadsPerBlock.x);

    harmonic_percussive_separation_int8_kernel<<<numBlocks, threadsPerBlock>>>(d_audio, d_harmonic, d_percussive, n, t, iterations);

    cudaMemcpy(harmonic, d_harmonic, t * sizeof(int8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_audio);
    cudaFree(d_harmonic);
    cudaFree(d_percussive);
}

}
