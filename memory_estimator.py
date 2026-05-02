"""
Peak Memory Usage Estimation (Section 3.5 of the HGNAS paper).

Implements the analytical memory model from Equations 6-9:
  - Sample  : M_sample = N_edges * 2 * U_index
  - Aggregate message construction: M_msg  = N_edges * 2 * L * U_k
  - Aggregate broadcasting        : M_broad = N * L * U_k
  - Combine  : M_com = N * L_out * U_k

Memory is tracked cumulatively through forward execution and
the global peak (highest M at any point) is returned.
"""

import config as C
from design_space import Architecture

# Precision constants (bytes)
U_INDEX = 8   # int64 edge indices
U_K     = 4   # float32 features

_MSG_MULT = {
    "source"   : 1,
    "target"   : 1,
    "relative" : 1,
    "src_rel"  : 2,
    "tgt_rel"  : 2,
    "euclidean": 1,
    "full"     : 3,
}


def estimate_peak_memory(
    architecture : Architecture,
    num_nodes    : int = C.NUM_POINTS,   # FIXED: was hardcoded to 1024
    k_neighbours : int = C.KNN_K,
    feature_dim  : int = C.HIDDEN_DIM,
) -> float:
    """
    Return estimated peak memory in MB for ONE sample (batch_size=1).
    """
    N = num_nodes
    L = feature_dim
    BYTES_PER_MB = 1024 * 1024

    param_bytes = 4 * (L * L) * 4 * len(architecture.positions)
    Mp = param_bytes / BYTES_PER_MB
    Md = (N * L * U_K) / BYTES_PER_MB

    M_current = Mp + Md
    M_peak    = M_current

    for pos_enc in architecture.positions:
        Ne = N * k_neighbours
        M_sample = (Ne * 2 * U_INDEX) / BYTES_PER_MB
        M_current += M_sample
        M_peak = max(M_peak, M_current)

        msg_type = pos_enc.agg_op[1]
        mult = _MSG_MULT.get(msg_type, 1)
        M_msg  = (Ne * 2 * L * mult * U_K) / BYTES_PER_MB
        M_broad = (N * L * U_K) / BYTES_PER_MB

        M_current += M_msg
        M_peak = max(M_peak, M_current)
        M_current = M_current - M_msg + M_broad

        L_out = pos_enc.combine_dim
        M_com = (N * L_out * U_K) / BYTES_PER_MB
        M_current += M_com
        M_peak = max(M_peak, M_current)

    return M_peak
