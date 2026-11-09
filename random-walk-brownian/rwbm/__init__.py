"""Random Walk to Brownian Motion — educational simulation toolkit."""

from .simulation import (
    simulate_random_walk,
    simulate_scaled_walk,
    simulate_scaled_walk_coupled,
    simulate_many_scaled,
)
from .statistics import empirical_mean_var, clt_samples, variance_along_time

__version__ = "1.0.0"

__all__ = [
    "simulate_random_walk",
    "simulate_scaled_walk",
    "simulate_scaled_walk_coupled",
    "simulate_many_scaled",
    "empirical_mean_var",
    "clt_samples",
    "variance_along_time",
]
