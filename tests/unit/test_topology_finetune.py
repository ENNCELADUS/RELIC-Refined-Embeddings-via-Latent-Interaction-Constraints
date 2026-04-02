"""Unit tests for graph-topology fine-tuning helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pytest
import torch
from src.topology.finetune_data import (
    build_pair_supervision_graph,
    filter_graph_to_embedding_index,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_training_subgraphs,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    build_symmetric_adjacency,
    compute_topology_losses,
    soft_graph_similarity_loss,
    soft_relative_density_loss,
)


def _write_embedding_cache(cache_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for protein_id, value in {
        "P1": 1.0,
        "P2": 2.0,
        "P3": 3.0,
        "P4": 4.0,
    }.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((2, 4), value, dtype=torch.float32), output_path)
        index[protein_id] = relative_path
    return index


def test_sample_training_subgraphs_supports_all_strategies() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("P1", "P2"),
            ("P2", "P3"),
            ("P3", "P4"),
            ("P4", "P5"),
            ("P5", "P6"),
        ]
    )

    for strategy in ["BFS", "DFS", "RANDOM_WALK", "mixed"]:
        sampled = sample_training_subgraphs(
            graph=graph,
            num_subgraphs=8,
            min_nodes=3,
            max_nodes=5,
            strategy=strategy,
            seed=13,
        )

        assert len(sampled) == 8
        assert all(3 <= len(node_ids) <= 5 for node_ids in sampled)
        assert all(set(node_ids).issubset(set(graph.nodes)) for node_ids in sampled)


def test_filter_graph_to_embedding_index_drops_unavailable_nodes() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])

    filtered = filter_graph_to_embedding_index(
        graph=graph,
        embedding_index={"P1": "p1.pt", "P2": "p2.pt"},
    )

    assert set(filtered.nodes) == {"P1", "P2"}
    assert set(filtered.edges) == {("P1", "P2")}


def test_build_pair_supervision_graph_uses_only_positive_pairs_and_keeps_isolated_nodes(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "human_BFS_split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump({"train": {"P1", "P2", "P3", "P4"}, "test": {"PX"}}, handle)
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "P1\tP2\t1\nP1\tP3\t0\nP2\tP3\t1\nPX\tP2\t1\nP4\tPY\t1\n",
        encoding="utf-8",
    )

    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    graph = build_pair_supervision_graph(pair_path=pair_path, node_ids=train_nodes)

    assert set(graph.nodes) == {"P1", "P2", "P3", "P4"}
    assert {tuple(sorted(edge)) for edge in graph.edges} == {("P1", "P2"), ("P2", "P3")}


def test_iter_subgraph_pair_chunks_materializes_all_labels(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    embedding_index = _write_embedding_cache(tmp_path / "cache")

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3"),
            cache_dir=tmp_path / "cache",
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=2,
        )
    )

    assert [tuple(chunk.pair_index_a.tolist()) for chunk in chunks] == [(0, 0), (1,)]
    assert [tuple(chunk.pair_index_b.tolist()) for chunk in chunks] == [(1, 2), (2,)]
    assert torch.cat([chunk.label for chunk in chunks], dim=0).tolist() == [1.0, 0.0, 1.0]
    assert chunks[0].emb_a.shape == torch.Size([2, 2, 4])
    assert chunks[0].emb_b.shape == torch.Size([2, 2, 4])


def test_soft_topology_losses_match_hard_metric_limits() -> None:
    target_adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pred_adjacency = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    gs_loss = soft_graph_similarity_loss(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
    )
    rd_loss = soft_relative_density_loss(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
    )

    assert gs_loss.item() == pytest.approx(0.5)
    assert rd_loss.item() == pytest.approx(0.0)


def test_compute_topology_losses_returns_weighted_total() -> None:
    pair_probabilities = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
    adjacency = build_symmetric_adjacency(
        num_nodes=3,
        pair_index_a=torch.tensor([0, 0, 1], dtype=torch.long),
        pair_index_b=torch.tensor([1, 2, 2], dtype=torch.long),
        pair_probabilities=pair_probabilities,
    )
    target = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    losses = compute_topology_losses(
        pred_adjacency=adjacency,
        target_adjacency=target,
        weights=TopologyLossWeights(alpha=0.5, beta=1.0, gamma=0.3, delta=0.2),
    )

    expected_total = (
        0.5 * losses["graph_similarity"]
        + losses["relative_density"]
        + 0.3 * losses["degree_mmd"]
        + 0.2 * losses["clustering_mmd"]
    )
    assert losses["total_topology"].item() == pytest.approx(expected_total.item())
