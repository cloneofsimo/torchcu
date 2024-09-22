import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SplinterEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_ids, position_ids):
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return word_embeddings + position_embeddings

class SplinterSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = num_attention_heads

    def forward(self, hidden_states, attention_mask):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(hidden_states.size(-1))
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

class SplinterLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.self_attention = SplinterSelfAttention(hidden_size, num_attention_heads)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.self_attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        return layer_output

def splinter_model(input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                   vocab_size: int = 30522, hidden_size: int = 768, 
                   num_attention_heads: int = 12, intermediate_size: int = 3072, 
                   max_position_embeddings: int = 512, num_hidden_layers: int = 12) -> torch.Tensor:
    embeddings = SplinterEmbeddings(vocab_size, hidden_size, max_position_embeddings)
    embedding_output = embeddings(input_ids, input_ids)

    encoder_layers = [SplinterLayer(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)]
    for i, layer in enumerate(encoder_layers):
        embedding_output = layer(embedding_output, attention_mask)

    return embedding_output



# function_signature
function_signature = {
    "name": "splinter_model",
    "inputs": [
        ((4, 4), torch.int64),
        ((4, 4), torch.int64),
    ],
    "outputs": [
        ((4, 4), torch.float32),
    ]
}