from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray


class BayesianSampler(Protocol):
    def update(self, data: NDArray) -> None: ...

    def reset(self) -> None: ...

    @property
    def trial(self) -> int: ...

    def sampling_mean(self, n: int) -> NDArray: ...

    def sampling_pred(self, n: int) -> NDArray: ...


class BinaryHyperParams(TypedDict):
    alpha: float
    beta: float


class BinaryBayesianSampler:
    def __init__(self, seed: int, alpha: float, beta: float):
        """ベイズ推定による二項分布のサンプリングを行うクラス

        Args:
            seed: _description_
            alpha: _description_
            beta: _description_
        """
        self.rng = np.random.default_rng(seed)
        self.init_alpha = alpha
        self.alpha = self.init_alpha
        self.init_beta = beta
        self.beta = self.init_beta
        self._trials = 0

    def update(self, data: NDArray) -> None:
        assert data.ndim == 1 and np.all(np.unique(data) == [0, 1])

        self._trials += len(data)
        successes = np.sum(data)
        failures = len(data) - successes

        self.alpha = self.alpha + float(successes)
        self.beta = self.beta + float(failures)

    def reset(self) -> None:
        self.alpha = self.init_alpha
        self.beta = self.init_beta
        self._trials = 0

    @property
    def p(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def trial(self) -> int:
        return self._trials

    def sampling_mean(self, n: int) -> NDArray:
        return self.rng.beta(self.alpha, self.beta, n)

    def sampling_pred(self, n: int) -> NDArray:
        return self.rng.binomial(1, self.p, n)


HyperParams = BinaryHyperParams
