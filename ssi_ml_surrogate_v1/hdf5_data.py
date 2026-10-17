import json
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

from .schema import GraphCase, ResponseCase


def _write_array(group, name, value):
    group.create_dataset(name, data=np.asarray(value), compression="lzf", shuffle=True)


def write_case(h5: h5py.File, case_id: str, graph: GraphCase, response: ResponseCase):
    graph.validate()
    response.validate(len(graph.time), len(graph.node_ids))
    root = h5.require_group(case_id)
    inputs = root.create_group("inputs")
    geometry = root.create_group("geometry")
    targets = root.create_group("targets")
    _write_array(inputs, "global_features", graph.global_features)
    _write_array(inputs, "ground_motion", graph.ground_motion)
    _write_array(inputs, "time", graph.time)
    _write_array(inputs, "node_features", graph.node_features)
    _write_array(inputs, "edge_index", graph.edge_index)
    _write_array(inputs, "edge_features", graph.edge_features)
    _write_array(geometry, "node_ids", graph.node_ids)
    _write_array(geometry, "pile_ids", graph.pile_ids.astype("S"))
    _write_array(targets, "node_response", response.node_response)
    _write_array(targets, "target_mask", response.target_mask)
    root.attrs["metadata_json"] = json.dumps({**graph.metadata, **response.metadata}, default=str)


def write_database(path: str, cases: Iterable[tuple], schema: Dict[str, object] | None = None):
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "SSI-ML-Surrogate-v1"
        h5.attrs["response_components"] = json.dumps(["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])
        if schema:
            h5.attrs["schema_json"] = json.dumps(schema, default=str)
        for case_id, graph, response in cases:
            write_case(h5, str(case_id), graph, response)


def write_input_database(path: str, cases: Iterable[tuple], schema: Dict[str, object] | None = None):
    """Write graph/input-only cases; these are not yet trainable targets."""
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "SSI-ML-Inputs-v1"
        h5.attrs["schema_json"] = json.dumps(schema or {}, default=str)
        for case_id, graph in cases:
            graph.validate()
            root = h5.require_group(str(case_id))
            inputs = root.create_group("inputs")
            geometry = root.create_group("geometry")
            for name in ("global_features", "ground_motion", "time", "node_features", "edge_index", "edge_features"):
                _write_array(inputs, name, getattr(graph, name))
            _write_array(geometry, "node_ids", graph.node_ids)
            _write_array(geometry, "pile_ids", graph.pile_ids.astype("S"))
            root.attrs["metadata_json"] = json.dumps(graph.metadata, default=str)


def read_graph_case(path: str, case_id: str) -> GraphCase:
    """Load an input-only case without requiring response targets."""
    with h5py.File(path, "r") as h5:
        root = h5[str(case_id)]
        inputs, geometry = root["inputs"], root["geometry"]
        graph = GraphCase(
            global_features=inputs["global_features"][:],
            ground_motion=inputs["ground_motion"][:], time=inputs["time"][:],
            node_features=inputs["node_features"][:], edge_index=inputs["edge_index"][:],
            edge_features=inputs["edge_features"][:], node_ids=geometry["node_ids"][:],
            pile_ids=geometry["pile_ids"][:].astype(str),
            metadata=json.loads(root.attrs.get("metadata_json", "{}")),
        )
        graph.validate()
        return graph


def list_cases(path: str) -> List[str]:
    with h5py.File(path, "r") as h5:
        return sorted(list(h5.keys()))


def read_peak_case(path: str, case_id: str):
    """Read full EQ input history and the six scalar first-node peak labels."""
    with h5py.File(path, "r") as h5:
        root=h5[case_id]; i=root["inputs"]; g=root["geometry"]
        graph=GraphCase(global_features=i["global_features"][:],ground_motion=i["ground_motion"][:],time=i["time"][:],
            node_features=i["node_features"][:],edge_index=i["edge_index"][:],edge_features=i["edge_features"][:],
            node_ids=g["node_ids"][:],pile_ids=g["pile_ids"][:].astype(str),metadata=json.loads(root.attrs.get("metadata_json","{}")))
        graph.validate()
        target=root["targets"]["peak_abs_response"][:].astype(np.float32)
        mask=root["targets"]["target_mask"][:].astype(np.float32)
        if target.shape!=(6,) or mask.shape!=(6,): raise ValueError(f"{case_id} should contain six static peak labels")
        return graph,target,mask


def read_case(path: str, case_id: str):
    with h5py.File(path, "r") as h5:
        root = h5[case_id]
        inputs = root["inputs"]
        geometry = root["geometry"]
        targets = root["targets"]
        graph = GraphCase(
            global_features=inputs["global_features"][:],
            ground_motion=inputs["ground_motion"][:],
            time=inputs["time"][:],
            node_features=inputs["node_features"][:],
            edge_index=inputs["edge_index"][:],
            edge_features=inputs["edge_features"][:],
            node_ids=geometry["node_ids"][:],
            pile_ids=geometry["pile_ids"][:].astype(str),
        )
        response = ResponseCase(
            node_response=targets["node_response"][:],
            target_mask=targets["target_mask"][:],
        )
        graph.validate(); response.validate(len(graph.time), len(graph.node_ids))
        return graph, response


def case_length(path: str, case_id: str) -> int:
    """Return the stored response-history length without loading the arrays."""
    with h5py.File(path, "r") as h5:
        return int(h5[case_id]["targets"]["node_response"].shape[0])


def read_case_window(path: str, case_id: str, start: int, end: int):
    """Load one temporal window while keeping the case graph static."""
    with h5py.File(path, "r") as h5:
        root = h5[case_id]
        inputs, geometry, targets = root["inputs"], root["geometry"], root["targets"]
        total = int(targets["node_response"].shape[0])
        start, end = max(0, int(start)), min(total, int(end))
        if end <= start:
            raise ValueError(f"Empty time window [{start}, {end}) for {case_id}")
        graph = GraphCase(
            global_features=inputs["global_features"][:],
            ground_motion=inputs["ground_motion"][start:end],
            time=inputs["time"][start:end],
            node_features=inputs["node_features"][:],
            edge_index=inputs["edge_index"][:],
            edge_features=inputs["edge_features"][:],
            node_ids=geometry["node_ids"][:],
            pile_ids=geometry["pile_ids"][:].astype(str),
            metadata=json.loads(root.attrs.get("metadata_json", "{}")),
        )
        response = ResponseCase(
            node_response=targets["node_response"][start:end],
            target_mask=targets["target_mask"][start:end],
        )
        graph.validate()
        response.validate(len(graph.time), len(graph.node_ids))
        return graph, response
