"""Fixed-vector Gaussian Process tools for first-node Ux/Uy peak labels."""
from __future__ import annotations

import numpy as np


def vectorize_arrays(global_features, node_features, edge_features, eq_sequence):
    """Create a fixed-length vector from one HDF5 graph case.

    Order: global inputs; node-channel mean/std/min/max; edge-channel
    mean/std/min/max; flattened ordered earthquake-window descriptors.
    """
    g = np.asarray(global_features, dtype=np.float64).reshape(-1)
    n = np.asarray(node_features, dtype=np.float64)
    e = np.asarray(edge_features, dtype=np.float64)
    q = np.asarray(eq_sequence, dtype=np.float64)
    parts = [g]
    for values in (n, e):
        stats = []
        for j in range(values.shape[1]):
            column = values[..., j].reshape(-1)
            if column.size:
                stats.extend((float(np.mean(column)), float(np.std(column)),
                              float(np.min(column)), float(np.max(column))))
            else:
                stats.extend((0.0, 0.0, 0.0, 0.0))
        parts.append(np.asarray(stats, dtype=np.float64))
    parts.append(q.reshape(-1))
    vector = np.concatenate(parts)
    if not np.isfinite(vector).all():
        raise ValueError("GPR inputs contain NaN or infinity")
    return vector


def vectorize_case(case):
    """Convert cross-validation case tuple into the GPR vector."""
    return vectorize_arrays(case[0], case[1], case[3], case[4])


def shapley_group_indices(groups, global_dim, node_dim, edge_dim, sequence_steps, sequence_dim):
    """Map named channels to vector columns; summaries/windows stay grouped."""
    node_start = global_dim
    edge_start = node_start + 4 * node_dim
    seq_start = edge_start + 4 * edge_dim
    result = []
    for key, column, _ in groups:
        if key == "global_features":
            indices = np.asarray([column], dtype=np.int64)
        elif key == "node_features":
            indices = node_start + 4 * column + np.arange(4, dtype=np.int64)
        elif key == "edge_features":
            indices = edge_start + 4 * column + np.arange(4, dtype=np.int64)
        elif key == "ground_motion":
            indices = seq_start + np.arange(sequence_steps, dtype=np.int64) * sequence_dim + column
        else:
            raise ValueError(f"Unsupported GPR feature group: {key}")
        result.append(indices)
    return result


def predict_bundle(bundle, x_raw):
    """Predict [Ux, Uy] means and standard deviations in metres."""
    x = np.asarray(x_raw, dtype=np.float64).reshape(1, -1)
    x_scaled = bundle["scaler"].transform(x)
    target_scale = np.asarray(bundle["target_scale"], dtype=np.float64)
    means = np.zeros(2, dtype=np.float64)
    stds = np.zeros(2, dtype=np.float64)
    for j, estimator in enumerate(bundle["estimators"]):
        mu, sigma = estimator.predict(x_scaled, return_std=True)
        means[j] = max(0.0, float(mu[0]) * target_scale[j])
        stds[j] = float(sigma[0]) * target_scale[j]
    return means.astype(np.float32), stds.astype(np.float32)


def sampled_shapley_gpr(bundle, x_raw, group_indices, permutations, rng):
    """Approximate grouped Shapley contributions in physical output units."""
    x = bundle["scaler"].transform(np.asarray(x_raw, dtype=np.float64).reshape(1, -1))[0]
    active = np.zeros_like(x)
    phi = np.zeros((len(group_indices), 2), dtype=np.float64)
    from .gpr_peak_v1 import predict_bundle
    for _ in range(permutations):
        active.fill(0.0)
        previous, _ = predict_bundle(bundle, bundle["scaler"].inverse_transform(active.reshape(1, -1))[0])
        for group_index in rng.permutation(len(group_indices)):
            indices = group_indices[int(group_index)]
            active[indices] = x[indices]
            raw_active = bundle["scaler"].inverse_transform(active.reshape(1, -1))[0]
            current, _ = predict_bundle(bundle, raw_active)
            phi[int(group_index)] += current - previous
            previous = current
    return (phi / float(permutations)).astype(np.float32)
