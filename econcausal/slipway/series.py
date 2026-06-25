import numpy as np

from econcausal.keel import FloatArray


def linear_sem_sample(
    weights: FloatArray, n_samples: int, noise_scale: float, seed: int
) -> FloatArray:
    rng = np.random.default_rng(seed)
    n = weights.shape[0]
    eye = np.eye(n, dtype=np.float64)
    noise = rng.normal(0.0, noise_scale, size=(n_samples, n)).astype(np.float64)
    propagator = np.linalg.inv(eye - weights)
    sample: FloatArray = noise @ propagator
    return sample


def standardize(series: FloatArray) -> FloatArray:
    centered = series - series.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    result: FloatArray = centered / scale
    return result
