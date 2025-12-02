# algos/random_walk.py
from __future__ import annotations
import numpy as np

from algos.base import Solver, SolverResult


class RandomWalkSolver(Solver):
    """
    매우 단순한 Random Walk 휴리스틱
    - 더 좋은 neighbor가 나오면 이동
    - 아니면 그대로 유지
    """

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

        for it in range(max_iter):
            neighbor = problem.get_neighbor(current_state)
            neighbor_cost, neighbor_metrics = problem.evaluate(neighbor)

            # 더 좋으면 이동
            if neighbor_cost < current_cost:
                current_state = neighbor
                current_cost = neighbor_cost
                current_metrics = neighbor_metrics

            # 글로벌 베스트 갱신
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                best_metrics = current_metrics.copy()

            history.append(
                {
                    "iter": it,
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[RW] iter={it:5d}, current_cost={current_cost:.4f}, "
                    f"best_cost={best_cost:.4f}, best_cov={best_metrics['coverage_ratio']:.4f}"
                )

        return SolverResult(
            best_state=best_state,
            best_cost=float(best_cost),
            best_metrics=best_metrics,
            history=history,
        )
