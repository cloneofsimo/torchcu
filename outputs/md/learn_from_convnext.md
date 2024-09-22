### ConvNext Model Implementation

#### ConvNext Embeddings

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNextEmbeddings(nn.Module):
    def __init__(self, config, num_channels):
        super(ConvNextEmbeddings, self).__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels=num_channels,
            out_channels=config.hidden_sizes[0],
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.layernorm = nn.LayerNorm(epsilon=1e-6)

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x
```

#### ConvNext Layer

```python
class ConvNextLayer(nn.Module):
    def __init__(self, config, dim, drop_path=0.0):
        super(ConvNextLayer, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding="same",
            groups=dim,
            bias=False,
        )
        self.layernorm = nn.LayerNorm(epsilon=1e-6)
        self.pwconv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=4 * dim,
            kernel_size=1,
            bias=False,
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(
            in_channels=4 * dim,
            out_channels=dim,
            kernel_size=1,
            bias=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        x = self.dwconv(hidden_states)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(-1, -2)
        x = x.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])
        x = hidden_states + self.drop_path(x)
        return x
```

#### ConvNext Stage

```python
class ConvNextStage(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super(ConvNextStage, self).__init__()
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [
                nn.LayerNorm(epsilon=1e-6),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                ),
            ]
        else:
            self.downsampling_layer = [nn.Identity()]
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = [
            ConvNextLayer(config, out_channels, drop_path=drop_path_rates[j])
            for j in range(depth)
        ]

    def forward(self, hidden_states):
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
```

#### ConvNext Encoder

```python
class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super(ConvNextEncoder, self).__init__()
        self.stages = []
        drop_path_rates = torch.linspace(0.0, config.drop_path_rate, sum(config.depths))
        drop_path_rates = torch.split(drop_path_rates, config.depths)
        drop_path_rates = [x.numpy().tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(self, hidden_states):
        for layer in self.stages:
            hidden_states = layer(hidden_states)
        return hidden_states
```

#### ConvNext Model

```python
class ConvNextModel(nn.Module):
    def __init__(self, config):
        super(ConvNextModel, self).__init__()
        self.embeddings = ConvNextEmbeddings(config, num_channels=3)
        self.encoder = ConvNextEncoder(config)
        self.layernorm = nn.LayerNorm(epsilon=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.layernorm(x)
        x = self.pooler(x)
        return x
```

#### ConvNext Image Processor

```python
import numpy as np

class ConvNextImageProcessor:
    def __init__(self, do_resize=True, size={"shortest_edge": 384}, crop_pct=224/256, resample=PILImageResampling.BILINEAR, do_rescale=True, rescale_factor=1/255, do_normalize=True, image_mean=IMAGENET_STANDARD_MEAN, image_std=IMAGENET_STANDARD_STD):
        self.do_resize = do_resize
        self.size = size
        self.crop_pct = crop_pct
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def resize(self, image, size, crop_pct, resample, input_data_format):
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" not in size:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        shortest_edge = size["shortest_edge"]

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=input_data_format
            )
            image = resize(
                image=image,
                size=resize_size,
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )
            # then crop to (shortest_edge, shortest_edge)
            return center_crop(
                image=image,
                size=(shortest_edge, shortest_edge),
                data_format=input_data_format,
                input_data_format=input_data_format,
            )
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return resize(
                image,
                size=(shortest_edge, shortest_edge),
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )

    def preprocess(self, images, do_resize=None, size=None, crop_pct=None, resample=None, do_rescale=None, rescale_factor=None, do_normalize=None, image_mean=None, image_std=None, return_tensors=None, data_format=None, input_data_format=None):
        do_resize = do_resize if do_resize is not None else self.do_resize
        crop_pct = crop_pct if crop_pct is not None else self.crop_pct
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
           