# algos/simulated_annealing.py
from __future__ import annotations
import numpy as np

from algos.base import Solver, SolverResult


class SimulatedAnnealingSolver(Solver):
    """
    Simulated Annealing 기반 탐색
    """

    def __init__(
        self,
        problem,
        rng: np.random.Generator | None = None,
        T_init: float = 1.0,
        alpha: float = 0.99,
        T_min: float = 1e-3,
    ):
        super().__init__(problem, rng)
        self.T_init = T_init
        self.alpha = alpha
        self.T_min = T_min

    def run(
        self,
        max_iter: int,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> SolverResult:
        problem = self.problem

        if init_state is None:
            current_state = problem.random_state()
        else:
            current_state = problem.clamp_state(init_state)

        current_cost, current_metrics = problem.evaluate(current_state)

        best_state = current_state.copy()
        best_cost = current_cost
        best_metrics = current_metrics.copy()

        history: list[dict] = []

        T = self.T_init

        for it in range(max_iter):
            neighbor = problem.get_neighbor(current_state)
            neighbor_cost, neighbor_metrics = problem.evaluate(neighbor)

            delta = neighbor_cost - current_cost

            accept = False
            if delta < 0:
                accept = True
            else:
                if T > 1e-12:
                    prob = np.exp(-delta / T)
                    if self.rng.random() < prob:
                        accept = True

            if accept:
                current_state = neighbor
                current_cost = neighbor_cost
                current_metrics = neighbor_metrics

            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                best_metrics = current_metrics.copy()

            history.append(
                {
                    "iter": it,
                    "T": float(T),
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[SA] iter={it:5d}, T={T:.4f}, current_cost={current_cost:.4f}, "
                    f"best_cost={best_cost:.4f}, best_cov={best_metrics['coverage_ratio']:.4f}"
                )

            # 온도 감소
            T = max(self.T_min, T * self.alpha)

        return SolverResult(
            best_state=best_state,
            best_cost=float(best_cost),
            best_metrics=best_metrics,
            history=history,
        )
