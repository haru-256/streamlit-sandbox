import numpy as np
from numpy.typing import NDArray


class Sampler:
    def __init__(self, seed: int, mean: float, std: float):
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.std = std

    def gen_data(self, n: int) -> NDArray[np.float64]:
        return self.rng.normal(self.mean, self.std, n)
