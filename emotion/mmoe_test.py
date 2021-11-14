from model import MMoE
import torch

batch_size = 1
hidden_dim = 2
seq_len = 3
x = torch.rand(batch_size, seq_len, hidden_dim)

mmoe = MMoE(hidden_dim, hidden_dim, hidden_dim, 4, 2)
mmoe(x)