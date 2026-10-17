from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

RESPONSE_COMPONENTS = ("Ux", "Uy", "Uz", "Rx", "Ry", "Rz")


@dataclass
class GraphCase:
    """Static graph plus time-dependent input motion for one simulation."""

    global_features: np.ndarray
    ground_motion: np.ndarray
    time: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_ids: np.ndarray
    pile_ids: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        self.global_features = np.asarray(self.global_features, dtype=np.float32).reshape(-1)
        self.ground_motion = np.asarray(self.ground_motion, dtype=np.float32)
        self.time = np.asarray(self.time, dtype=np.float32).reshape(-1)
        self.node_features = np.asarray(self.node_features, dtype=np.float32)
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        self.edge_features = np.asarray(self.edge_features, dtype=np.float32)
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64).reshape(-1)
        self.pile_ids = np.asarray(self.pile_ids).reshape(-1)
        if self.ground_motion.ndim != 2 or len(self.time) != len(self.ground_motion):
            raise ValueError("ground_motion must be [T,A] and time must have length T")
        if self.node_features.ndim != 2 or len(self.node_ids) != len(self.node_features):
            raise ValueError("node_features must be [N,F] and match node_ids")
        if len(self.pile_ids) != len(self.node_ids):
            raise ValueError("pile_ids must match node_ids")
        if self.edge_index.shape[0] != 2 or self.edge_index.shape[1] != len(self.edge_features):
            raise ValueError("edge_index must be [2,E] and edge_features must be [E,Fe]")
        if self.edge_index.size and int(self.edge_index.max()) >= len(self.node_ids):
            raise ValueError("edge_index refers to a node outside this graph")


@dataclass
class ResponseCase:
    """Existing OpenSees target response for one simulation."""

    node_response: np.ndarray
    target_mask: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self, n_time: int, n_nodes: int) -> None:
        self.node_response = np.asarray(self.node_response, dtype=np.float32)
        if self.node_response.ndim != 3 or self.node_response.shape[:2] != (n_time, n_nodes):
            raise ValueError("node_response must be [T,N,C] and match graph dimensions")
        if self.target_mask is None:
            self.target_mask = np.isfinite(self.node_response).astype(np.float32)
        else:
            self.target_mask = np.asarray(self.target_mask, dtype=np.float32)
        if self.target_mask.shape != self.node_response.shape:
            raise ValueError("target_mask must match node_response")
