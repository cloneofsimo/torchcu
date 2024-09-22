
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x

def torch_model_l1_isin_function(input_tensor: torch.Tensor, model: MyModel, labels: torch.Tensor) -> torch.Tensor:
    """
    This function runs a model, computes L1 loss and checks if predicted class is in a list of labels.
    """
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1)

    # L1 loss between predicted and target (one-hot encoded)
    target = one_hot(labels, num_classes=output.shape[1])
    loss = torch.nn.L1Loss()(output, target)

    # Check if predicted class is in a list of labels
    isin = torch.isin(predicted_class, labels)

    return loss, isin.float()

function_signature = {
    "name": "torch_model_l1_isin_function",
    "inputs": [
        ((10, 10), torch.float32),
        (MyModel, None),
        ((10,), torch.int64)
    ],
    "outputs": [
        ((), torch.float32),
        ((10,), torch.float32)
    ]
}
