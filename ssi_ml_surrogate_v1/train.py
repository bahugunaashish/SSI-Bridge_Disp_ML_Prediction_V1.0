"""Shared helpers and single-model peak training entry point."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .hdf5_data import list_cases, read_graph_case, read_peak_case
from .models import build_model

FEATURE_KEYS = ("global_features", "ground_motion", "node_features", "edge_features")


def tensor_case(graph, normalizer=None):
    result = {
        key: torch.from_numpy(np.asarray(getattr(graph, key)).copy())
        for key in ("global_features", "ground_motion", "time", "node_features", "edge_index", "edge_features")
    }
    if normalizer:
        for key in FEATURE_KEYS:
            stats = normalizer[key]
            mean = torch.as_tensor(stats["mean"], dtype=result[key].dtype)
            scale = torch.as_tensor(stats["scale"], dtype=result[key].dtype)
            result[key] = (result[key] - mean) / scale
    return result


def fit_input_normalizer(database, cases):
    """Fit per-feature standardization from training cases only."""
    sums, squares, counts = {}, {}, {}
    for case_id in cases:
        graph = read_graph_case(database, case_id)
        for key in FEATURE_KEYS:
            values = np.asarray(getattr(graph, key), dtype=np.float64)
            values = values.reshape(-1, values.shape[-1])
            sums[key] = sums.get(key, np.zeros(values.shape[1], dtype=np.float64)) + values.sum(axis=0)
            squares[key] = squares.get(key, np.zeros(values.shape[1], dtype=np.float64)) + np.square(values).sum(axis=0)
            counts[key] = counts.get(key, 0) + len(values)
    normalizer = {}
    for key in FEATURE_KEYS:
        mean = sums[key] / max(counts[key], 1)
        variance = np.maximum(squares[key] / max(counts[key], 1) - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1.0
        normalizer[key] = {"mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist()}
    return normalizer


def fit_target_scale(database, cases):
    """Scale each nonnegative peak output by its training-set RMS."""
    values = np.asarray([read_peak_case(database, case_id)[1] for case_id in cases], dtype=np.float64)
    scale = np.sqrt(np.mean(np.square(values), axis=0))
    return np.maximum(scale, 1e-8).astype(np.float32).tolist()


def first_node_index(graph):
    ids = graph.metadata.get("recorded_node_ids", [])
    if not ids:
        raise ValueError("Simulation input has no recorded_node_ids metadata")
    indices = np.flatnonzero(np.asarray(graph.node_ids, dtype=np.int64) == int(ids[0]))
    if len(indices) != 1:
        raise ValueError("First recorder node is not present exactly once in graph inputs")
    return int(indices[0])


def load_dims(database):
    import h5py
    case_id = list_cases(database)[0]
    with h5py.File(database, "r") as h5:
        case = h5[case_id]
        if "peak_abs_response" not in case["targets"]:
            raise ValueError("This must be a peak-label HDF5 database; run the peak TXT importer.")
        inputs = case["inputs"]
        return (inputs["node_features"].shape[1], inputs["edge_features"].shape[1],
                inputs["global_features"].shape[0], inputs["ground_motion"].shape[1], 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cases = list_cases(args.database)
    np.random.default_rng(args.seed).shuffle(cases)
    dims = load_dims(args.database)
    normalizer = fit_input_normalizer(args.database, cases)
    target_scale = np.asarray(fit_target_scale(args.database, cases), dtype=np.float32)
    target_scale_tensor = torch.from_numpy(target_scale)
    model = build_model(args.model, node_dim=dims[0], edge_dim=dims[1], global_dim=dims[2],
                        motion_dim=dims[3], response_dim=6, hidden=args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for case_id in cases:
            graph, target, mask = read_peak_case(args.database, case_id)
            pred = model(tensor_case(graph, normalizer))[first_node_index(graph)]
            target_tensor = torch.from_numpy(target) / target_scale_tensor
            mask_tensor = torch.from_numpy(mask)
            loss = (torch.nn.functional.smooth_l1_loss(pred, target_tensor, reduction="none") * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = float(np.mean(losses))
        print(f"epoch {epoch:03d}: train_normalized={score:.6g}", flush=True)
        if score < best:
            best = score
            checkpoint = Path(args.checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "model_name": args.model, "dims": dims,
                        "hidden": args.hidden, "input_normalizer": normalizer,
                        "target_scale": target_scale.tolist()}, checkpoint)


if __name__ == "__main__":
    main()
