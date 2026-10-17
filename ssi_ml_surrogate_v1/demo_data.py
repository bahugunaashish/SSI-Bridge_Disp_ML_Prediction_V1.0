"""Generate synthetic variable-size cases for smoke testing the package."""

import argparse
import numpy as np

from .hdf5_data import write_database
from .schema import GraphCase, ResponseCase


def make_case(i: int):
    rng = np.random.default_rng(i)
    n_nodes = 12 + (i % 4) * 3
    n_steps = 64 + (i % 3) * 8
    t = np.arange(n_steps, dtype=np.float32) * 0.01
    gm = rng.normal(0, 0.03 + 0.005 * i, size=(n_steps, 3)).astype(np.float32)
    gm[:, 0] += 0.05 * np.sin(2 * np.pi * 1.2 * t)
    z = np.linspace(0, -20, n_nodes)
    x = np.tile(np.linspace(-4, 4, 6), int(np.ceil(n_nodes / 6)))[:n_nodes]
    pile_type = (np.arange(n_nodes) % 2).astype(np.float32)
    node_features = np.column_stack([x, np.zeros(n_nodes), z, np.abs(z), pile_type]).astype(np.float32)
    edges = np.array([[j, j + 1] for j in range(n_nodes - 1)], dtype=np.int64).T
    edge_features = np.column_stack([
        np.ones(edges.shape[1]),
        np.linalg.norm(node_features[edges[0], :3] - node_features[edges[1], :3], axis=1),
        np.zeros(edges.shape[1]),
    ]).astype(np.float32)
    response = np.zeros((n_steps, n_nodes, 6), dtype=np.float32)
    response[:, :, 0:3] = np.cumsum(gm[:, None, :] * (1 + np.abs(z)[None, :, None] / 20), axis=0) * 0.01
    response[:, :, 3:6] = response[:, :, 0:3] * 0.01
    graph = GraphCase(
        global_features=np.array([3, 1.2, 20, 0.4, np.sin(i), np.cos(i)], dtype=np.float32),
        ground_motion=gm, time=t, node_features=node_features,
        edge_index=edges, edge_features=edge_features,
        node_ids=np.arange(1000 + i * 100, 1000 + i * 100 + n_nodes),
        pile_ids=np.array([f"P{j // 3 + 1}" for j in range(n_nodes)]),
        metadata={"case_index": i},
    )
    return f"case_{i:04d}", graph, ResponseCase(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="demo_training.h5")
    p.add_argument("--cases", type=int, default=12)
    a = p.parse_args()
    write_database(a.output, (make_case(i) for i in range(a.cases)))
    print(f"Wrote {a.cases} cases to {a.output}")


if __name__ == "__main__":
    main()
