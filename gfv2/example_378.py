
import torch
import torch.nn as nn
from torch.autograd import Function
from scipy.optimize import linear_sum_assignment
import numpy as np


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        Computes the contrastive loss between two sets of features.

        Args:
            features1 (torch.Tensor): Features from the first set, shape (batch_size, feature_dim).
            features2 (torch.Tensor): Features from the second set, shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: The contrastive loss.
        """
        # Calculate similarity matrix
        similarity = features1 @ features2.T
        similarity /= self.temperature

        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(features1.size(0), device=features1.device)

        # Apply mask for positive pairs
        mask = torch.eye(features1.size(0), dtype=torch.bool, device=features1.device)
        similarity = similarity[~mask].view(features1.size(0), -1)
        labels = labels[~mask].view(features1.size(0), -1)

        # Compute loss using cross-entropy
        loss = nn.CrossEntropyLoss()(similarity, labels)

        return loss


class WassersteinDistance(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        """
        Computes the Wasserstein distance between two input tensors.

        Args:
            ctx: Context object to save inputs and outputs for backward pass.
            input1 (torch.Tensor): First input tensor, shape (batch_size, feature_dim).
            input2 (torch.Tensor): Second input tensor, shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: The Wasserstein distance between input1 and input2.
        """
        ctx.save_for_backward(input1, input2)
        batch_size = input1.size(0)

        # Reshape inputs to (batch_size, 1, feature_dim)
        input1 = input1.view(batch_size, 1, -1)
        input2 = input2.view(batch_size, 1, -1)

        # Calculate cost matrix
        cost_matrix = torch.cdist(input1, input2, p=2)

        # Use Hungarian algorithm for optimal matching
        _, indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        indices = torch.tensor(indices, device=input1.device)

        # Gather matched elements from input2
        matched_input2 = torch.gather(input2, 1, indices.unsqueeze(1))

        # Calculate Wasserstein distance as mean of squared distances
        distance = torch.mean((input1 - matched_input2) ** 2)
        return distance

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradients for the Wasserstein distance.

        Args:
            ctx: Context object containing saved inputs and outputs.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. output.

        Returns:
            tuple: Gradients for input1 and input2.
        """
        input1, input2 = ctx.saved_tensors
        batch_size = input1.size(0)

        # Reshape inputs to (batch_size, 1, feature_dim)
        input1 = input1.view(batch_size, 1, -1)
        input2 = input2.view(batch_size, 1, -1)

        # Calculate cost matrix
        cost_matrix = torch.cdist(input1, input2, p=2)

        # Use Hungarian algorithm for optimal matching
        _, indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        indices = torch.tensor(indices, device=input1.device)

        # Gather matched elements from input2
        matched_input2 = torch.gather(input2, 1, indices.unsqueeze(1))

        # Calculate gradients for input1 and input2
        grad_input1 = 2 * (input1 - matched_input2) * grad_output
        grad_input2 = -grad_input1
        
        return grad_input1.view(batch_size, -1), grad_input2.view(batch_size, -1)


class MyFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, input4, input5):
        """
        Performs a series of operations:
          1. Calculates the determinant of input1
          2. Computes the Wasserstein distance between input2 and input3
          3. Reshapes input4
          4. Calculates the contrastive loss between input5 and the reshaped input4
          5. Returns the determinant, Wasserstein distance, and contrastive loss as a list

        Args:
            ctx: Context object to save inputs and outputs for backward pass.
            input1 (torch.Tensor): Input tensor for determinant calculation, shape (batch_size, feature_dim).
            input2 (torch.Tensor): First input tensor for Wasserstein distance, shape (batch_size, feature_dim).
            input3 (torch.Tensor): Second input tensor for Wasserstein distance, shape (batch_size, feature_dim).
            input4 (torch.Tensor): Input tensor for reshaping, shape (batch_size, feature_dim).
            input5 (torch.Tensor): First input tensor for contrastive loss, shape (batch_size, feature_dim).

        Returns:
            list: A list containing the determinant, Wasserstein distance, and contrastive loss.
        """
        ctx.save_for_backward(input1, input2, input3, input4, input5)

        # Calculate the determinant of input1
        det = torch.det(input1)

        # Calculate the Wasserstein distance between input2 and input3
        wasserstein_distance = WassersteinDistance.apply(input2, input3)

        # Reshape input4
        reshaped_input4 = input4.view(-1, 16)  # Assume you want to reshape to (batch_size, 16)

        # Calculate the contrastive loss between input5 and reshaped input4
        contrastive_loss = ContrastiveLoss()(input5, reshaped_input4)

        # Return the results as a list
        return [det, wasserstein_distance, contrastive_loss]

    @staticmethod
    def backward(ctx, grad_det, grad_wasserstein, grad_contrastive):
        """
        Computes the gradients for the forward operations.

        Args:
            ctx: Context object containing saved inputs and outputs.
            grad_det (torch.Tensor): Gradient of the loss w.r.t. determinant.
            grad_wasserstein (torch.Tensor): Gradient of the loss w.r.t. Wasserstein distance.
            grad_contrastive (torch.Tensor): Gradient of the loss w.r.t. contrastive loss.

        Returns:
            tuple: Gradients for input1, input2, input3, input4, and input5.
        """
        input1, input2, input3, input4, input5 = ctx.saved_tensors

        # Calculate gradients for each input
        grad_input1 = torch.linalg.inv(input1) * grad_det  # Gradient for determinant
        grad_input2, grad_input3 = WassersteinDistance.backward(ctx, grad_wasserstein)  # Gradients for Wasserstein distance
        grad_input4 = ContrastiveLoss.backward(ctx, grad_contrastive)  # Gradients for contrastive loss
        grad_input5 = ContrastiveLoss.backward(ctx, grad_contrastive)  # Gradient for input5 (contrastive loss)

        return grad_input1, grad_input2, grad_input3, grad_input4, grad_input5

