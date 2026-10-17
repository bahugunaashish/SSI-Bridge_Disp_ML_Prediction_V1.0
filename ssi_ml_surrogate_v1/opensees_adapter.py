"""OpenSeesPy feature-extraction interface.

The real user model builder will be connected here later. It must create only
static model metadata and must not run the dynamic analysis during inference.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from .schema import GraphCase


DT_OPENSEES = 0.005
EDGE_TYPE_NAMES = (
    "dispBeamColumn", "elasticBeamColumn", "zeroLength", "twoNodeLink",
    "forceBeamColumn", "SSPbrick", "other",
)


def graph_from_opensees(op, case: Dict, node_tags=None, element_tags=None,
                        pile_node_tags=None, bottom_node_tags=None,
                        response_node_tags=None, response_element_tags=None,
                        dt: float = DT_OPENSEES) -> GraphCase:
    """Read a just-built OpenSees model and case inputs without solving it.

    Ground-motion scaling and horizontal component directions match the setup
    in ``main_v4.py``. Element connectivity is expanded into undirected graph
    edges; element ids and exact element type strings are kept in metadata.
    """
    if case is None:
        raise ValueError("feature_only requires a simulation-matrix case")
    node_tags = list(op.getNodeTags() if node_tags is None else node_tags)
    element_tags = list(op.getEleTags() if element_tags is None else element_tags)
    if not node_tags:
        raise ValueError("The model build produced no OpenSees nodes")

    node_ids = np.asarray(node_tags, dtype=np.int64)
    tag_to_index = {int(tag): i for i, tag in enumerate(node_ids)}
    coords = np.asarray([op.nodeCoord(int(tag))[:3] for tag in node_ids], dtype=np.float32)
    pairs = []
    edge_features = []
    edge_element_ids = []
    edge_element_types = []
    degree = np.zeros(len(node_ids), dtype=np.float32)
    for ele_tag in element_tags:
        connected = [int(tag) for tag in op.eleNodes(int(ele_tag)) if int(tag) in tag_to_index]
        if len(connected) < 2:
            continue
        try:
            ele_type = str(op.eleType(int(ele_tag)))
        except Exception:
            ele_type = "other"
        type_index = EDGE_TYPE_NAMES.index(ele_type) if ele_type in EDGE_TYPE_NAMES[:-1] else len(EDGE_TYPE_NAMES) - 1
        type_features = np.zeros(len(EDGE_TYPE_NAMES), dtype=np.float32)
        type_features[type_index] = 1.0
        for a in range(len(connected)):
            for b in range(a + 1, len(connected)):
                i, j = tag_to_index[connected[a]], tag_to_index[connected[b]]
                length = float(np.linalg.norm(coords[i] - coords[j]))
                for src, dst in ((i, j), (j, i)):
                    pairs.append((src, dst))
                    edge_features.append(np.r_[length, len(connected), type_features])
                    edge_element_ids.append(int(ele_tag))
                    edge_element_types.append(ele_type)
                    degree[src] += 1.0
    if not pairs:
        raise ValueError("No two-node-or-more element connectivity was found")

    pile_nodes = set(map(int, [] if pile_node_tags is None else pile_node_tags))
    bottom_nodes = set(map(int, [] if bottom_node_tags is None else bottom_node_tags))
    is_pile_or_column = np.asarray([int(tag) in pile_nodes for tag in node_ids], dtype=np.float32)
    is_pile_bottom = np.asarray([int(tag) in bottom_nodes for tag in node_ids], dtype=np.float32)
    node_features = np.column_stack((coords, degree, is_pile_or_column, is_pile_bottom)).astype(np.float32)
    edge_index = np.asarray(pairs, dtype=np.int64).T
    edge_features = np.asarray(edge_features, dtype=np.float32)

    eq_path = str(case["EQPath"])
    raw_motion = np.asarray(np.loadtxt(eq_path), dtype=np.float64)
    if raw_motion.ndim != 1:
        raise ValueError(f"Expected one ground-motion series in {eq_path}; got shape {raw_motion.shape}")
    if raw_motion.size < 2 or not np.isfinite(raw_motion).all():
        raise ValueError(f"Ground-motion file is empty or contains non-finite values: {eq_path}")
    current_pga = float(np.max(np.abs(raw_motion)))
    if current_pga <= 0:
        raise ValueError(f"Ground-motion record has zero PGA: {eq_path}")
    target_pga = float(case["target_pga"])
    angle = math.radians(float(case["incident_angle_deg"]))
    scale = 9.81 * target_pga / current_pga
    # main_v4 applies sin(angle) in global X and cos(angle) in global Y.
    gm = np.column_stack((raw_motion * scale * math.sin(angle),
                          raw_motion * scale * math.cos(angle))).astype(np.float32)
    time = np.arange(len(raw_motion), dtype=np.float32) * float(dt)

    soil = case.get("soil_layers") or []
    def weighted(key):
        thickness = np.asarray([max(0.0, float(x["top"]) - float(x["bottom"])) for x in soil])
        values = np.asarray([float(x.get(key, 0.0)) for x in soil])
        return float(np.average(values, weights=thickness)) if thickness.sum() else 0.0
    global_features = np.asarray([
        float(case["H"]), float(case["D"]), float(case["pl"]), float(case["fc"]),
        float(case["fy"]), float(case["number_of_spans"]), float(case["span_length"]),
        float(case["number_of_piles"]), float(case["pile_spacing"]), float(case["L_P"]),
        target_pga, math.sin(angle), math.cos(angle), float(len(soil)),
        weighted("gamma"), weighted("phi"), weighted("su"), weighted("G"),
        float(dt),
    ], dtype=np.float32)
    graph = GraphCase(
        global_features=global_features, ground_motion=gm, time=time,
        node_features=node_features, edge_index=edge_index,
        edge_features=edge_features, node_ids=node_ids,
        # No stable pile-group map is exposed for every node; don't guess it.
        pile_ids=np.full(len(node_ids), "unknown", dtype="U16"),
        metadata={
            "simulation_no": case.get("simulation_no"), "earthquake": case.get("EQ"),
            "earthquake_file": eq_path, "ground_motion_units": "m/s^2",
            "raw_record_pga_g": current_pga,
            "target_pga_g": target_pga,
            "pga_scale_factor": target_pga / current_pga,
            "incident_angle_deg": float(case["incident_angle_deg"]),
            "ground_motion_components": ["accel_global_x", "accel_global_y"],
            "component_projection": "X=sin(angle), Y=cos(angle), matching main_v4.py",
            "ground_motion_dt_s": float(dt), "analysis_stop_fraction": 0.8,
            "global_feature_names": [
                "column_height_m", "diameter_m", "steel_ratio_percent", "fc_Pa", "fy_Pa",
                "number_of_spans", "span_length_m", "piles_per_bent", "pile_spacing_m",
                "pile_embedment_m", "target_pga_g", "sin_incidence_angle", "cos_incidence_angle",
                "soil_layer_count", "thickness_weighted_gamma", "thickness_weighted_phi_deg",
                "thickness_weighted_su_Pa", "thickness_weighted_G_Pa",
                "ground_motion_dt_s",
            ],
            "node_feature_names": ["x_m", "y_m", "z_m", "graph_degree", "is_pile_or_column_node", "is_pile_bottom_node"],
            "edge_feature_names": ["length_m", "element_node_count", *[f"type_{x}" for x in EDGE_TYPE_NAMES]],
            "recorded_node_ids": [int(x) for x in ([] if response_node_tags is None else response_node_tags)],
            "recorded_element_ids": [int(x) for x in ([] if response_element_tags is None else response_element_tags)],
            "element_ids_per_directed_edge": edge_element_ids,
            "element_types_per_directed_edge": edge_element_types,
        },
    )
    graph.validate()
    return graph


class OpenSeesPyFeatureExtractor:
    """Adapter for a user-supplied OpenSeesPy static model builder.

    `model_builder(parameters)` must return a dictionary with keys matching the
    GraphCase constructor. It may call OpenSeesPy node/element/material APIs,
    but should stop before dynamic analysis and recorders are run.
    """

    def __init__(self, model_builder: Callable[[Dict], Dict]):
        self.model_builder = model_builder

    def extract(self, parameters: Dict) -> GraphCase:
        raw = self.model_builder(parameters)
        required = {
            "global_features", "ground_motion", "time", "node_features",
            "edge_index", "edge_features", "node_ids", "pile_ids",
        }
        missing = required.difference(raw)
        if missing:
            raise KeyError(f"OpenSeesPy model builder did not return: {sorted(missing)}")
        graph = GraphCase(**{k: raw[k] for k in required}, metadata=raw.get("metadata", {}))
        graph.validate()
        return graph


class DummyOpenSeesBuilder:
    """Deterministic stand-in until the real OpenSeesPy script is attached."""

    def __call__(self, parameters: Dict) -> Dict:
        n_nodes = int(parameters.get("n_nodes", 12))
        n_steps = int(parameters.get("n_steps", 80))
        seed = int(parameters.get("seed", 0))
        rng = np.random.default_rng(seed)
        t = np.arange(n_steps, dtype=np.float32) * float(parameters.get("dt", 0.01))
        gm = rng.normal(0.0, 0.02, size=(n_steps, 3)).astype(np.float32)
        x = np.linspace(-3.0, 3.0, n_nodes)
        z = np.linspace(0.0, -20.0, n_nodes)
        node_features = np.column_stack([x, np.zeros(n_nodes), z, np.abs(z), np.ones(n_nodes)]).astype(np.float32)
        edges = np.array([[i, i + 1] for i in range(n_nodes - 1)], dtype=np.int64).T
        edge_features = np.ones((edges.shape[1], 3), dtype=np.float32)
        return {
            "global_features": np.array([
                parameters.get("n_spans", 3), parameters.get("pile_diameter", 1.2),
                parameters.get("pile_depth", 20.0), parameters.get("pga", 0.4),
                np.sin(np.deg2rad(parameters.get("angle_deg", 0.0))),
                np.cos(np.deg2rad(parameters.get("angle_deg", 0.0))),
            ], dtype=np.float32),
            "ground_motion": gm,
            "time": t,
            "node_features": node_features,
            "edge_index": edges,
            "edge_features": edge_features,
            "node_ids": np.arange(1001, 1001 + n_nodes),
            "pile_ids": np.array([f"EB-1P{(i % 6) + 1}" for i in range(n_nodes)]),
            "metadata": {"source": "dummy_openSeesPy_builder"},
        }


def load_parameters(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
