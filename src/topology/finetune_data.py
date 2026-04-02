"""Subgraph sampling and pair materialization for topology fine-tuning."""

from __future__ import annotations

import pickle
import random
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch
from torch.nn.utils.rnn import pad_sequence

from src.embed import load_cached_embedding

SUPPORTED_SAMPLING_STRATEGIES = {"BFS", "DFS", "RANDOM_WALK", "MIXED"}


@dataclass(frozen=True)
class SubgraphPairChunk:
    """Chunked pair batch materialized from one sampled training subgraph."""

    nodes: tuple[str, ...]
    emb_a: torch.Tensor
    emb_b: torch.Tensor
    len_a: torch.Tensor
    len_b: torch.Tensor
    label: torch.Tensor
    pair_index_a: torch.Tensor
    pair_index_b: torch.Tensor


def filter_graph_to_embedding_index(
    *,
    graph: nx.Graph,
    embedding_index: Mapping[str, str],
) -> nx.Graph:
    """Return the node-induced subgraph restricted to embeddable proteins."""
    available_nodes = [node for node in graph.nodes if node in embedding_index]
    if not available_nodes:
        raise ValueError("No train-graph nodes have cached embeddings")
    return graph.subgraph(available_nodes).copy()


def load_split_node_ids(*, split_path: Path, split_name: str) -> set[str]:
    """Load protein IDs for one split from a PRING ``*_split.pkl`` file."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Split pickle must contain a dictionary: {split_path}")

    split_ids = payload.get(split_name)
    if not isinstance(split_ids, set) or not all(isinstance(item, str) for item in split_ids):
        raise ValueError(f"Split '{split_name}' in {split_path} must be a set of protein IDs")
    return set(split_ids)


def build_pair_supervision_graph(
    *,
    pair_path: Path,
    node_ids: set[str],
) -> nx.Graph:
    """Build a supervision graph from in-split positive pairs and a fixed node universe."""
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair dataset not found: {pair_path}")

    graph = nx.Graph()
    graph.add_nodes_from(sorted(node_ids))
    with pair_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
                continue
            try:
                label = int(float(parts[2]))
            except ValueError:
                continue
            if label <= 0:
                continue
            if parts[0] not in node_ids or parts[1] not in node_ids:
                continue
            graph.add_edge(parts[0], parts[1])
    return graph


def _sample_target_size(
    *,
    graph: nx.Graph,
    min_nodes: int,
    max_nodes: int,
    rng: random.Random,
) -> int:
    """Sample a valid target node count for one subgraph."""
    if min_nodes <= 1:
        raise ValueError("min_nodes must be greater than 1")
    if max_nodes < min_nodes:
        raise ValueError("max_nodes must be >= min_nodes")
    graph_size = graph.number_of_nodes()
    if graph_size < min_nodes:
        raise ValueError(
            f"Train graph is too small for subgraph sampling: {graph_size} < {min_nodes}"
        )
    upper_bound = min(max_nodes, graph_size)
    return rng.randint(min_nodes, upper_bound)


def _fallback_complete_nodes(
    *,
    nodes_in_order: list[str],
    graph_nodes: Sequence[str],
    target_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Pad a partially explored node set with random unseen nodes."""
    if len(nodes_in_order) >= target_size:
        return tuple(nodes_in_order[:target_size])

    seen = set(nodes_in_order)
    remaining = [node for node in graph_nodes if node not in seen]
    rng.shuffle(remaining)
    nodes_in_order.extend(remaining[: target_size - len(nodes_in_order)])
    return tuple(nodes_in_order)


def _sample_bfs_nodes(graph: nx.Graph, target_size: int, rng: random.Random) -> tuple[str, ...]:
    """Sample one node set by randomized breadth-first expansion."""
    graph_nodes = list(graph.nodes)
    queue: deque[str] = deque([rng.choice(graph_nodes)])
    visited: set[str] = set()
    nodes_in_order: list[str] = []

    while queue and len(nodes_in_order) < target_size:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        nodes_in_order.append(node)
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)

        if not queue and len(nodes_in_order) < target_size:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            if unseen:
                queue.append(rng.choice(unseen))

    return _fallback_complete_nodes(
        nodes_in_order=nodes_in_order,
        graph_nodes=graph_nodes,
        target_size=target_size,
        rng=rng,
    )


def _sample_dfs_nodes(graph: nx.Graph, target_size: int, rng: random.Random) -> tuple[str, ...]:
    """Sample one node set by randomized depth-first expansion."""
    graph_nodes = list(graph.nodes)
    stack = [rng.choice(graph_nodes)]
    visited: set[str] = set()
    nodes_in_order: list[str] = []

    while stack and len(nodes_in_order) < target_size:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        nodes_in_order.append(node)
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)

        if not stack and len(nodes_in_order) < target_size:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            if unseen:
                stack.append(rng.choice(unseen))

    return _fallback_complete_nodes(
        nodes_in_order=nodes_in_order,
        graph_nodes=graph_nodes,
        target_size=target_size,
        rng=rng,
    )


def _sample_random_walk_nodes(
    graph: nx.Graph,
    target_size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Sample one node set by randomized walk with restart fallback."""
    graph_nodes = list(graph.nodes)
    current = rng.choice(graph_nodes)
    visited = {current}
    nodes_in_order = [current]
    max_steps = max(target_size * 20, 100)

    for _ in range(max_steps):
        if len(nodes_in_order) >= target_size:
            break
        neighbors = list(graph.neighbors(current))
        if neighbors:
            current = rng.choice(neighbors)
        else:
            unseen = [candidate for candidate in graph_nodes if candidate not in visited]
            current = rng.choice(unseen) if unseen else rng.choice(graph_nodes)
        if current not in visited:
            visited.add(current)
            nodes_in_order.append(current)

    return _fallback_complete_nodes(
        nodes_in_order=nodes_in_order,
        graph_nodes=graph_nodes,
        target_size=target_size,
        rng=rng,
    )


def sample_training_subgraphs(
    *,
    graph: nx.Graph,
    num_subgraphs: int,
    min_nodes: int,
    max_nodes: int,
    strategy: str,
    seed: int,
) -> list[tuple[str, ...]]:
    """Sample node-induced training subgraphs from a PRING train graph."""
    if num_subgraphs <= 0:
        raise ValueError("num_subgraphs must be positive")

    normalized_strategy = strategy.upper()
    if normalized_strategy not in SUPPORTED_SAMPLING_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_SAMPLING_STRATEGIES))
        raise ValueError(f"Unsupported subgraph sampling strategy: {strategy} ({supported})")

    rng = random.Random(seed)
    sampled_nodes: list[tuple[str, ...]] = []
    for _ in range(num_subgraphs):
        selected_strategy = normalized_strategy
        if normalized_strategy == "MIXED":
            selected_strategy = rng.choice(["BFS", "DFS", "RANDOM_WALK"])
        target_size = _sample_target_size(
            graph=graph,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            rng=rng,
        )
        if selected_strategy == "BFS":
            sampled_nodes.append(_sample_bfs_nodes(graph, target_size, rng))
        elif selected_strategy == "DFS":
            sampled_nodes.append(_sample_dfs_nodes(graph, target_size, rng))
        else:
            sampled_nodes.append(_sample_random_walk_nodes(graph, target_size, rng))
    return sampled_nodes


def _subgraph_pair_tuples(
    *,
    graph: nx.Graph,
    nodes: tuple[str, ...],
) -> list[tuple[int, int, str, str, float]]:
    """Return all upper-triangle node pairs with graph labels."""
    pair_rows: list[tuple[int, int, str, str, float]] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            protein_b = nodes[index_b]
            label = 1.0 if graph.has_edge(protein_a, protein_b) else 0.0
            pair_rows.append((index_a, index_b, protein_a, protein_b, label))
    if not pair_rows:
        raise ValueError("A sampled subgraph must contain at least one node pair")
    return pair_rows


def iter_subgraph_pair_chunks(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
) -> Iterator[SubgraphPairChunk]:
    """Yield all within-subgraph pairs as padded mini-batches."""
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")

    node_tuple = tuple(nodes)
    if len(node_tuple) < 2:
        raise ValueError("A sampled subgraph must contain at least two nodes")

    embeddings = {
        protein_id: load_cached_embedding(
            cache_dir=cache_dir,
            index=embedding_index,
            protein_id=protein_id,
            expected_input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        )
        for protein_id in node_tuple
    }
    pair_rows = _subgraph_pair_tuples(graph=graph, nodes=node_tuple)

    for chunk_start in range(0, len(pair_rows), pair_batch_size):
        rows = pair_rows[chunk_start : chunk_start + pair_batch_size]
        emb_a = pad_sequence(
            [embeddings[protein_a] for _, _, protein_a, _, _ in rows],
            batch_first=True,
        )
        emb_b = pad_sequence(
            [embeddings[protein_b] for _, _, _, protein_b, _ in rows],
            batch_first=True,
        )
        yield SubgraphPairChunk(
            nodes=node_tuple,
            emb_a=emb_a,
            emb_b=emb_b,
            len_a=torch.tensor(
                [embeddings[protein_a].size(0) for _, _, protein_a, _, _ in rows],
                dtype=torch.long,
            ),
            len_b=torch.tensor(
                [embeddings[protein_b].size(0) for _, _, _, protein_b, _ in rows],
                dtype=torch.long,
            ),
            label=torch.tensor([label for _, _, _, _, label in rows], dtype=torch.float32),
            pair_index_a=torch.tensor([index_a for index_a, _, _, _, _ in rows], dtype=torch.long),
            pair_index_b=torch.tensor([index_b for _, index_b, _, _, _ in rows], dtype=torch.long),
        )
