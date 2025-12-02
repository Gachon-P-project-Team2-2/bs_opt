# algos/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from core.model import BaseStationProblem


@dataclass
class SolverResult:
    best_state: np.ndarray
    best_cost: float
    best_metrics: dict
    history: list[dict] = field(default_factory=list)


class Solver:
    """
    모든 탐색 알고리즘이 상속받는 베이스 클래스
    """

    def __init__(
        self,
        problem: BaseStationProblem,
        rng: np.random.Generator | None = None,
    ):
        self.problem = problem
        self.rng = rng or np.random.default_rng()

    def run(
        self,
        max_iter: int,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> SolverResult:
        """
        자식 클래스에서 구현해야 하는 인터페이스
        """
        raise NotImplementedError
