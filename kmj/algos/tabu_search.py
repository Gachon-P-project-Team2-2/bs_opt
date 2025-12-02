# algos/tabu_search.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

from algos.base import Solver, SolverResult


@dataclass
class TabuParams:
    tenure: int = 10              # Tabu 리스트에 머무르는 기간
    n_candidates: int = 30        # 한 iteration에 평가할 neighbor 개수
    max_no_improve: int = 100     # 개선이 없는 iteration이 이만큼 쌓이면 중단


class TabuSearchSolver(Solver):
    """
    간단한 Tabu Search 구현
    - move를 (bs_index, new_x, new_y)로 정의
    - 최근 사용된 move를 Tabu 리스트에 저장하고, 그 move는 일정 기간 동안 사용 불가
    - Aspiration: Tabu move라도 글로벌 베스트보다 개선되면 허용
    """

    def __init__(
        self,
        problem,
        rng: np.random.Generator | None = None,
        params: TabuParams | None = None,
    ):
        super().__init__(problem, rng)
        self.params = params or TabuParams()

    def _generate_neighbor_with_move(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        하나의 neighbor와 해당 move 정보를 생성
        move: (idx, new_x, new_y)
        """
        s = state.copy()
        n_bs = s.shape[0]
        idx = self.rng.integers(0, n_bs)

        # [-1, 0, +1] 이동
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)

        new_x = int(np.clip(s[idx, 0] + dx, 0, self.problem.width - 1))
        new_y = int(np.clip(s[idx, 1] + dy, 0, self.problem.height - 1))

        s[idx, 0] = new_x
        s[idx, 1] = new_y

        move = (idx, new_x, new_y)
        return s, move

    def run(
        self,
        max_iter: int,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> SolverResult:
        problem = self.problem
        p = self.params

        if init_state is None:
            current_state = problem.random_state()
        else:
            current_state = problem.clamp_state(init_state)

        current_cost, current_metrics = problem.evaluate(current_state)
        best_state = current_state.copy()
        best_cost = current_cost
        best_metrics = current_metrics.copy()

        # Tabu 리스트: move -> 남은 tenure
        tabu_dict: Dict[Tuple[int, int, int], int] = {}

        history: List[dict] = []
        no_improve_count = 0

        for it in range(max_iter):
            candidate_states: List[np.ndarray] = []
            candidate_moves: List[Tuple[int, int, int]] = []
            candidate_costs: List[float] = []
            candidate_metrics: List[dict] = []

            # 여러 후보 neighbor 생성
            for _ in range(p.n_candidates):
                neighbor, move = self._generate_neighbor_with_move(current_state)

                cost, metrics = problem.evaluate(neighbor)
                candidate_states.append(neighbor)
                candidate_moves.append(move)
                candidate_costs.append(cost)
                candidate_metrics.append(metrics)

            # 후보 중에서 Tabu가 아닌(또는 Aspiration 조건 만족) 것 중 best 선택
            best_candidate_idx = None
            best_candidate_cost = float("inf")
            best_candidate_metrics = None
            best_candidate_state = None
            best_candidate_move = None

            for i, (state_i, move_i, cost_i, metrics_i) in enumerate(
                zip(candidate_states, candidate_moves, candidate_costs, candidate_metrics)
            ):
                in_tabu = move_i in tabu_dict and tabu_dict[move_i] > 0

                # Aspiration 기준: Tabu더라도 글로벌 best보다 좋으면 허용
                if in_tabu and cost_i >= best_cost:
                    continue

                if cost_i < best_candidate_cost:
                    best_candidate_cost = cost_i
                    best_candidate_idx = i
                    best_candidate_state = state_i
                    best_candidate_metrics = metrics_i
                    best_candidate_move = move_i

            if best_candidate_idx is None:
                # 모든 후보가 Tabu인데 Aspiration도 못 만족 → 그래도 하나는 선택 (가장 좋은 것)
                i = int(np.argmin(candidate_costs))
                best_candidate_state = candidate_states[i]
                best_candidate_cost = candidate_costs[i]
                best_candidate_metrics = candidate_metrics[i]
                best_candidate_move = candidate_moves[i]

            # 선택된 move를 적용
            current_state = best_candidate_state
            current_cost = best_candidate_cost
            current_metrics = best_candidate_metrics

            # Tabu 리스트 업데이트 (tenure 초기화)
            if best_candidate_move is not None:
                tabu_dict[best_candidate_move] = p.tenure

            # 글로벌 베스트 갱신
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                best_metrics = current_metrics.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Tabu 남은 기간 감소
            to_delete = []
            for move, t in tabu_dict.items():
                tabu_dict[move] = t - 1
                if tabu_dict[move] <= 0:
                    to_delete.append(move)
            for move in to_delete:
                del tabu_dict[move]

            history.append(
                {
                    "iter": it,
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                    "tabu_size": len(tabu_dict),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[Tabu] iter={it:5d}, current_cost={current_cost:.4f}, "
                    f"best_cost={best_cost:.4f}, best_cov={best_metrics['coverage_ratio']:.4f}, "
                    f"tabu_size={len(tabu_dict)}"
                )

            if no_improve_count >= p.max_no_improve:
                if verbose:
                    print(f"[Tabu] Early stop: no improvement for {p.max_no_improve} iterations")
                break

        return SolverResult(
            best_state=best_state,
            best_cost=float(best_cost),
            best_metrics=best_metrics,
            history=history,
        )
