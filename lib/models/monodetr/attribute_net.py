import torch
import torch.nn.functional as F
from torch import nn


class AttributeNet(nn.Module):
    """Two-layer MLP used to learn an attribute-specific query feature."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        ])

    def forward(self, x):
        return self.layers[1](F.relu(self.layers[0](x)))


def groupwise_random_route(chain, parallel, group_num, chain_prob=0.5, mask=None):
    """Select one depth branch for every query group.

    The returned mask can be reused by every decoder layer so that auxiliary
    losses supervise the same route as the final layer.
    """
    if chain.shape != parallel.shape or chain.ndim != 3:
        raise ValueError("chain and parallel must have the same [B, Q, D] shape")
    if not 0.0 <= chain_prob <= 1.0:
        raise ValueError("chain_prob must be in [0, 1]")

    batch_size, num_queries, feature_dim = chain.shape
    if group_num <= 0 or num_queries % group_num != 0:
        raise ValueError(
            f"num_queries ({num_queries}) must be divisible by group_num ({group_num})")

    queries_per_group = num_queries // group_num
    chain_grouped = chain.reshape(
        batch_size, group_num, queries_per_group, feature_dim)
    parallel_grouped = parallel.reshape_as(chain_grouped)

    if mask is None:
        mask = torch.rand(
            batch_size, group_num, 1, 1, device=chain.device) < chain_prob
    if mask.shape != (batch_size, group_num, 1, 1):
        raise ValueError(
            "route mask must have shape "
            f"{(batch_size, group_num, 1, 1)}, got {tuple(mask.shape)}")

    routed = torch.where(mask, chain_grouped, parallel_grouped)
    return routed.reshape_as(chain), mask
