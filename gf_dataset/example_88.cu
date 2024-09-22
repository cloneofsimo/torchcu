
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void triplet_margin_loss_with_interpolation_kernel(const float *anchor, const float *positive, const float *negative,
                                                              float *loss, int batch_size, int feature_dim, float margin, float p) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float dist_pos = 0.0f;
        float dist_neg = 0.0f;
        float dist_interp = 0.0f;

        for (int i = 0; i < feature_dim; i++) {
            float a = anchor[idx * feature_dim + i];
            float p = positive[idx * feature_dim + i];
            float n = negative[idx * feature_dim + i];

            dist_pos += pow(abs(a - p), p);
            dist_neg += pow(abs(a - n), p);
            dist_interp += pow(abs(a - (p + n) / 2.0f), p);
        }

        dist_pos = pow(dist_pos, 1.0f / p);
        dist_neg = pow(dist_neg, 1.0f / p);
        dist_interp = pow(dist_interp, 1.0f / p);

        loss[idx] = max(0.0f, dist_pos - dist_neg + margin) + max(0.0f, dist_interp - dist_neg + margin);
    }
}

extern "C" {

void triplet_margin_loss_with_interpolation(int num_args, ...) {
    va_list args;
    va_start(args, num_args);

    const float *anchor = va_arg(args, const float*);
    int anchor_dim0 = va_arg(args, int);
    int anchor_dim1 = va_arg(args, int);

    const float *positive = va_arg(args, const float*);
    int positive_dim0 = va_arg(args, int);
    int positive_dim1 = va_arg(args, int);

    const float *negative = va_arg(args, const float*);
    int negative_dim0 = va_arg(args, int);
    int negative_dim1 = va_arg(args, int);

    float *loss = va_arg(args, float*);

    float margin = va_arg(args, float);
    float p = va_arg(args, float);

    va_end(args);

    if (anchor_dim0 != positive_dim0 || anchor_dim0 != negative_dim0 || anchor_dim1 != positive_dim1 || anchor_dim1 != negative_dim1) {
        return; // Handle error: dimensions mismatch
    }

    int batch_size = anchor_dim0;
    int feature_dim = anchor_dim1;

    float *d_anchor, *d_positive, *d_negative, *d_loss;

    cudaMalloc(&d_anchor, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_positive, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_negative, batch_size * feature_dim * sizeof(float));
    cudaMalloc(&d_loss, batch_size * sizeof(float));

    cudaMemcpy(d_anchor, anchor, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, positive, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, negative, batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice);

    triplet_margin_loss_with_interpolation_kernel<<<(batch_size + 255) / 256, 256>>>(
        d_anchor, d_positive, d_negative, d_loss, batch_size, feature_dim, margin, p
    );

    cudaMemcpy(loss, d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);
}

}
