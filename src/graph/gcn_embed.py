"""Graph embedding via a Graph Autoencoder (GAE, Kipf & Welling 2016) with a
2-layer GCNConv encoder -- unlike node2vec_embed.py, this consumes each
filter's raw statistic vector as the GCN's input feature (not just graph
topology), and unlike diffusion_embed.py, the propagation weights between
layers are learned rather than fixed.

Deliberately not node2vec/DeepWalk-family: those factorize a matrix built
purely from random-walk co-occurrence statistics, which is provably a
function of graph topology alone (TADW's own motivation is proving DeepWalk
is exactly that factorization, then bolting a content factor onto it) and
never looks at node content. A GAE instead trains by reconstructing the
graph's edges from embeddings produced by message-passing over node
features, so both content and topology are load-bearing to what it learns.

No random walks, so this sidesteps the memory-scaling problems documented in
node2vec_embed.py entirely -- training runs directly on the edge list (one
negative-sampling call per epoch, via `torch_geometric.utils.negative_sampling`
inside `GAE.recon_loss`), not on generated walk sequences.

Uses `torch_geometric` core only (`GCNConv`, `GAE`) -- deliberately NOT
`torch_scatter`/`torch_sparse`/`pyg_lib`, which are optional compiled-kernel
accelerators PyG can run without. That distinction matters here: missing
prebuilt wheels for those exact packages (for our torch/CUDA combo) is what
ruled out `torch_geometric.nn.Node2Vec` in the first place. GCNConv's base
implementation has no such compiled-extension dependency.

Caveat: `edge_weight` is used by `GCNConv`'s message passing (so it does
shape what the encoder learns), but PyG's built-in `GAE.recon_loss` is plain
edge/non-edge binary cross-entropy -- it does not fold edge weight into the
reconstruction target itself. So "weighted" here means edge strength shapes
*how features propagate*, not that the decoder is trained to reproduce exact
similarity values.
"""

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GCNConv

from src.embedding.activation_matrix import ActivationMatrix
from src.graph.similarity_graph import _flatten_filters


@dataclass
class GCNEmbedConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 100
    lr: float = 0.01
    # Weighted (cosine-similarity strength shapes message passing) vs
    # unweighted (every kept edge treated equally) encoding.
    use_edge_weights: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 42
    # See _flatten_filters (similarity_graph.py) -- z-scores each of the 7
    # stats independently before this becomes the GCN's input feature vector,
    # so the encoder's first linear layer isn't dominated by whichever raw
    # stat has the largest magnitude.
    standardize_stats: bool = True


class _GCNEncoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(h, edge_index, edge_weight)


def _graph_to_edge_tensors(graph: nx.Graph, use_edge_weights: bool, device: torch.device):
    num_nodes = graph.number_of_nodes()
    weight_attr = "weight" if use_edge_weights else None
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(num_nodes), weight=weight_attr, format="coo")
    edge_index = torch.tensor(np.vstack([adjacency.row, adjacency.col]), dtype=torch.long, device=device)
    edge_weight = (
        torch.tensor(adjacency.data, dtype=torch.float32, device=device) if use_edge_weights else None
    )
    return edge_index, edge_weight


def compute_gcn_embeddings(
    graph: nx.Graph, activation_matrix: ActivationMatrix, config: GCNEmbedConfig = None
) -> np.ndarray:
    """Returns (num_filters, embed_dim), row i is filter i's GAE embedding."""
    config = config or GCNEmbedConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)
    device = torch.device(config.device)

    x = torch.tensor(
        _flatten_filters(activation_matrix.representation, standardize=config.standardize_stats),
        dtype=torch.float32,
        device=device,
    )
    edge_index, edge_weight = _graph_to_edge_tensors(graph, config.use_edge_weights, device)

    model = GAE(_GCNEncoder(x.shape[1], config.hidden_dim, config.embed_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index, edge_weight)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_weight)
    return z.cpu().numpy().astype(np.float32)
