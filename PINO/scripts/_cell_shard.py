"""Shared cell -> GPU shard mapping for the PINO Darcy phase sweep (8-GPU layout).

Within each n bucket (8r × 3s = 24 cells), 3 cells are assigned to each of the
8 shards, giving each shard exactly 3 cells per n -- perfectly balanced.
"""

# r grid for the Darcy phase plot (hash fallback for other r, e.g. curriculum intermediates)
R_LIST = [1, 2, 4, 6, 8, 10]
S_LIST = [0, 1, 2]


def cell_shard(r, n, s, n_shards=8, f=1.0, tau=3.0, alpha=2.0, a_low=3.0):
    """Returns shard ID (0..n_shards-1) for cell (r, n, s, f [, tau, alpha, a_low]).

    When ``tau, alpha, a_low`` match the piececonst defaults (3, 2, 3), hashing matches
    the historical layout (coefficients ignored). Otherwise coefficients participate in
    the shard key.
    """
    if abs(float(tau) - 3.0) < 1e-12 and abs(float(alpha) - 2.0) < 1e-12 and abs(float(a_low) - 3.0) < 1e-12:
        try:
            r_idx = R_LIST.index(int(r))
            s_idx = S_LIST.index(int(s))
        except ValueError:
            return abs(hash((float(r), float(f), int(n), int(s)))) % n_shards
        return abs(hash((r_idx, float(f), int(n), s_idx))) % n_shards
    return (
        abs(
            hash(
                (
                    float(r),
                    float(f),
                    int(n),
                    int(s),
                    float(tau),
                    float(alpha),
                    float(a_low),
                )
            )
        )
        % n_shards
    )
