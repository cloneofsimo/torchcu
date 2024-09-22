import torch
import torch.nn.functional as F

def complex_task(x: torch.Tensor, kernel_size: int, padding: int, max_pool_kernel_size: int):
    """
    This function performs a series of operations on an input tensor, including:
    1. 2D Convolution
    2. Batch Normalization
    3. ReLU activation
    4. 2D Max Pooling

    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the convolutional kernel
        padding: Padding to apply to the input before convolution
        max_pool_kernel_size: Size of the max pooling kernel

    Returns:
        Output tensor after the series of operations
    """

    # Check if CUDA is available and move the tensor to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    # 2D Convolution
    conv_layer = torch.nn.Conv2d(in_channels=x.shape[1], out_channels=16, kernel_size=kernel_size, padding=padding)
    conv_layer = conv_layer.to(device)
    x = conv_layer(x)

    # Batch Normalization
    bn_layer = torch.nn.BatchNorm2d(num_features=16)
    bn_layer = bn_layer.to(device)
    x = bn_layer(x)

    # ReLU activation
    x = F.relu(x)

    # 2D Max Pooling
    x = F.max_pool2d(x, kernel_size=max_pool_kernel_size)

    return x

function_signature = {
    'name': 'complex_task',
    'inputs': [
        ((64, 3, 32, 32), torch.float32),
        ((1, 5), int),  # kernel_size (max value 5)
        ((1, 2), int),  # padding (max value 2)
        ((1, 4), int)   # max_pool_kernel_size (max value 4)
    ],
    'outputs': [
        ((64, 16, 16, 16), torch.float32)  # Output shape is now specified
    ],
}
