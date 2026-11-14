"""
Random streams of an experiment's iterations, derived from one master seed.

Iteration i draws from child i of SeedSequence(master_seed), i.e. SeedSequence(master_seed, spawn_key=(i,)),
which is what SeedSequence(master_seed).spawn(n)[i] returns. Each child can be built on its own, so iterations
can run in any order, in any process, and raising the number of iterations leaves the earlier ones unchanged.
The children are independent streams, which consecutive integer seeds are not guaranteed to be.

Without a master seed (None) the iteration number is the seed itself, which is how the experiments run before
master seeds were introduced were drawn; it keeps their saved results reproducible.
"""

import numpy as np


def iteration_seed(master_seed: int | None, iteration: int) -> int | np.random.SeedSequence:
    """What to seed np.random.default_rng with for one iteration of an experiment."""
    if master_seed is None:
        return iteration
    return np.random.SeedSequence(master_seed, spawn_key=(iteration,))
