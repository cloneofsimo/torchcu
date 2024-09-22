
import torch

def torch_similarity_rank_fold_function(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, dim: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the cosine similarity between two input tensors, filters for values below a threshold,
    computes the matrix rank of the filtered tensor, and folds the result along a specified dimension.
    """
    similarity = torch.nn.functional.cosine_similarity(input_tensor1, input_tensor2, dim=dim)
    filtered_similarity = torch.where(similarity < 0.5, torch.zeros_like(similarity), similarity)
    rank = torch.linalg.matrix_rank(filtered_similarity)
    folded_result = torch.nn.functional.fold(rank, output_size=(k, k), kernel_size=(k, k), stride=(k, k))
    return similarity, folded_result

function_signature = {
    "name": "torch_similarity_rank_fold_function",
    "inputs": [
        ((4, 4), torch.float32),
        ((4, 4), torch.float32),
        ((), torch.int32),
        ((), torch.int32),
    ],
    "outputs": [
        ((4, 4), torch.float32),
        ((1, 1), torch.int64),
    ]
}
