# simple_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.ffn(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_scores = self.gating(x)  # (batch, num_experts)
        topk_vals, topk_idx = torch.topk(gate_scores, self.top_k, dim=-1)

        expert_outputs = []
        for i in range(self.top_k):
            expert = self.experts[topk_idx[:, i]]
            weight = topk_vals[:, i].unsqueeze(-1)
            output = expert(x) * weight
            expert_outputs.append(output)

        return sum(expert_outputs)

# Test
if __name__ == "__main__":
    moe = MoELayer(input_dim=16, hidden_dim=32, num_experts=4, top_k=2)
    x = torch.randn(8, 16)  # batch of 8 tokens
    out = moe(x)
    print(out.shape)  # should be (8, 16)
