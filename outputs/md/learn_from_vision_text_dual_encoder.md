### VisionTextDualEncoderModel
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VisionTextDualEncoderModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim):
        super(VisionTextDualEncoderModel, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.projection_dim = projection_dim
        self.visual_projection = nn.Linear(vision_model.config.hidden_size, projection_dim)
        self.text_projection = nn.Linear(text_model.config.hidden_size, projection_dim)
        self.logit_scale = nn.Parameter(torch.ones((1,), dtype=torch.float32))

    def forward(self, input_ids, pixel_values, attention_mask=None, return_loss=False):
        vision_outputs = self.vision_model(pixel_values)
        text_outputs = self.text_model(input_ids)

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        image_embeds = self.visual_projection(image_embeds)
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = torch.exp(self.logit_scale)
        logits_per_text = torch.matmul(text_embeds, image_embeds.transpose(-1, -2)) * logit_scale
        logits_per_image = torch.transpose(logits_per_text, -1, -2)

        loss = None
        if return_loss:
            loss = F.mse_loss(logits_per_text, torch.zeros_like(logits_per_text))

        return loss, logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs
```

### ContrastiveLoss
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, logits):
        return F.mse_loss(logits, torch.zeros_like(logits))
```

### ClipLoss
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()

    def forward(self, similarity):
        caption_loss = F.mse_loss(similarity, torch.zeros_like(similarity))
        image_loss = F.mse_loss(torch.transpose(similarity, -1, -2), torch.zeros_like(similarity))
        return (caption_loss + image_loss) / 2.0
```