### ConvNextV2 Embeddings Function
```python
import torch

class ConvNextV2Embeddings(torch.nn.Module):
    def __init__(self, num_channels, hidden_sizes, patch_size):
        super().__init__()
        self.patch_embeddings = torch.nn.Conv2d(num_channels, hidden_sizes[0], kernel_size=patch_size, stride=patch_size)
        self.layernorm = torch.nn.LayerNorm(hidden_sizes[0], eps=1e-6)

    def forward(self, pixel_values):
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError("Make sure that the channel dimension of the pixel values match with the one set in the configuration.")
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings
```

### ConvNextV2 Drop Path Function
```python
import torch

class ConvNextV2DropPath(torch.nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        return drop_path(hidden_states, self.drop_prob, self.training)
```

### ConvNextV2 Stage Function
```python
import torch

class ConvNextV2Stage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depth, drop_path_rates):
        super().__init__()
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = torch.nn.Sequential(
                torch.nn.LayerNorm(in_channels, eps=1e-6),
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = torch.nn.Identity()
        self.layers = torch.nn.Sequential(
            *[ConvNextV2Layer(torch.nn.Linear(out_channels, 4*out_channels), out_channels, drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states):
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states
```

### ConvNextV2 Encoder Function
```python
import torch

class ConvNextV2Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = torch.nn.ModuleList()
        drop_path_rates = torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        drop_path_rates = [x.tolist() for x in drop_path_rates]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return torch.nn.ModuleList([hidden_states, all_hidden_states])
```

### ConvNextV2 Model Function
```python
import torch

class ConvNextV2Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = ConvNextV2Embeddings(config.num_channels, config.hidden_sizes, config.patch_size)
        self.encoder = ConvNextV2Encoder(config)
        self.layernorm = torch.nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return torch.nn.ModuleList([last_hidden_state, pooled_output, encoder_outputs[1]])
```