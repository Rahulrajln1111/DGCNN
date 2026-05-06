"""
Fine-grained hierarchical design space for HGNAS (Section 3.3).

Each supernet position holds ONE choice from each of:
  - Sample  : knn | random
  - Aggregate : (agg_type, msg_type)
  - Combine : output_dim
  - Connect : identity | skip

Function Space  → aggregator type, message type, combine dim, sample type
Operation Space → which operations appear at each position
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import config as C


# ── Encoding helpers ──────────────────────────────────────────────────────────

def sample_idx_to_name(idx: int) -> str:
    return C.SAMPLE_OPS[idx % len(C.SAMPLE_OPS)]

def agg_idx_to_name(idx: int) -> Tuple[str, str]:
    """Returns (aggregator_type, message_type)."""
    agg  = C.AGGREGATE_TYPES[idx % len(C.AGGREGATE_TYPES)]
    msg  = C.MESSAGE_TYPES[(idx // len(C.AGGREGATE_TYPES)) % len(C.MESSAGE_TYPES)]
    return agg, msg

def combine_idx_to_dim(idx: int) -> int:
    return C.COMBINE_DIMS[idx % len(C.COMBINE_DIMS)]

def connect_idx_to_name(idx: int) -> str:
    return C.CONNECT_OPS[idx % len(C.CONNECT_OPS)]


# ── Architecture encoding ─────────────────────────────────────────────────────

@dataclass
class PositionEncoding:
    """One-hot style encoding for a single supernet position."""
    sample_idx  : int   # index into SAMPLE_OPS
    agg_idx     : int   # compound index for (agg_type × msg_type)
    combine_idx : int   # index into COMBINE_DIMS
    connect_idx : int   # index into CONNECT_OPS

    @property
    def sample_op(self)  -> str:             return sample_idx_to_name(self.sample_idx)
    @property
    def agg_op(self)     -> Tuple[str, str]: return agg_idx_to_name(self.agg_idx)
    @property
    def combine_dim(self)-> int:             return combine_idx_to_dim(self.combine_idx)
    @property
    def connect_op(self) -> str:             return connect_idx_to_name(self.connect_idx)

    def to_list(self):
        return [self.sample_idx, self.agg_idx, self.combine_idx, self.connect_idx]

    @staticmethod
    def from_list(lst):
        return PositionEncoding(*lst)

    def to_onehot(self) -> List[int]:
        """
        One-hot vector for predictor node features (Section 3.5).
        Layout:  [sample(2)] + [agg_type(4)] + [msg_type(7)] + [combine(6)] + [connect(2)]
        Total = 2 + 4 + 7 + 6 + 2 = 21 dims
        """
        def onehot(idx, size):
            v = [0] * size
            v[idx % size] = 1
            return v

        agg_t_idx = self.agg_idx % len(C.AGGREGATE_TYPES)
        msg_t_idx = (self.agg_idx // len(C.AGGREGATE_TYPES)) % len(C.MESSAGE_TYPES)

        return (
            onehot(self.sample_idx,  len(C.SAMPLE_OPS))       +
            onehot(agg_t_idx,        len(C.AGGREGATE_TYPES))  +
            onehot(msg_t_idx,        len(C.MESSAGE_TYPES))    +
            onehot(self.combine_idx, len(C.COMBINE_DIMS))     +
            onehot(self.connect_idx, len(C.CONNECT_OPS))
        )

    ONEHOT_DIM = (
        len(C.SAMPLE_OPS) +
        len(C.AGGREGATE_TYPES) +
        len(C.MESSAGE_TYPES) +
        len(C.COMBINE_DIMS) +
        len(C.CONNECT_OPS)
    )


@dataclass
class Architecture:
    """Full architecture: one PositionEncoding per supernet position."""
    positions: List[PositionEncoding]

    def __len__(self):
        return len(self.positions)

    def to_list(self):
        return [p.to_list() for p in self.positions]

    @staticmethod
    def from_list(lst):
        return Architecture([PositionEncoding.from_list(p) for p in lst])

    @staticmethod
    def random(num_positions: int = C.NUM_POSITIONS) -> "Architecture":
        positions = []
        for _ in range(num_positions):
            positions.append(PositionEncoding(
                sample_idx  = random.randint(0, len(C.SAMPLE_OPS) - 1),
                agg_idx     = random.randint(0, len(C.AGGREGATE_TYPES) * len(C.MESSAGE_TYPES) - 1),
                combine_idx = random.randint(0, len(C.COMBINE_DIMS) - 1),
                connect_idx = random.randint(0, len(C.CONNECT_OPS) - 1),
            ))
        return Architecture(positions)


# ── Design space utilities ────────────────────────────────────────────────────

class DesignSpace:
    """
    The hierarchical design space from Section 3.3.

    Function Space  ← aggregator, message type, combine dim, sample type
    Operation Space ← operation type at each position (connect/aggregate/combine/sample)

    For simplicity in this implementation we treat all four choices
    per position as the 'operation' and use the sharing scheme
    (upper half / lower half) described in the paper for Stage 1.
    """

    def __init__(self, num_positions: int = C.NUM_POSITIONS):
        self.N = num_positions
        self.half = num_positions // 2

    # Total choices per position
    @property
    def num_sample_choices(self):  return len(C.SAMPLE_OPS)
    @property
    def num_agg_choices(self):     return len(C.AGGREGATE_TYPES) * len(C.MESSAGE_TYPES)
    @property
    def num_combine_choices(self): return len(C.COMBINE_DIMS)
    @property
    def num_connect_choices(self): return len(C.CONNECT_OPS)

    def random_architecture(self) -> Architecture:
        return Architecture.random(self.N)

    def mutate(self, arch: Architecture, mutation_prob: float = C.EA_MUTATION_PROB) -> Architecture:
        """Mutate one or more positions of an architecture."""
        new_positions = []
        for pos in arch.positions:
            if random.random() < mutation_prob:
                new_positions.append(PositionEncoding(
                    sample_idx  = random.randint(0, self.num_sample_choices - 1),
                    agg_idx     = random.randint(0, self.num_agg_choices - 1),
                    combine_idx = random.randint(0, self.num_combine_choices - 1),
                    connect_idx = random.randint(0, self.num_connect_choices - 1),
                ))
            else:
                new_positions.append(PositionEncoding(*pos.to_list()))
        return Architecture(new_positions)

    def crossover(self, arch_a: Architecture, arch_b: Architecture) -> Architecture:
        """Single-point crossover between two architectures."""
        cut = random.randint(1, self.N - 1)
        positions = arch_a.positions[:cut] + arch_b.positions[cut:]
        return Architecture(positions)
