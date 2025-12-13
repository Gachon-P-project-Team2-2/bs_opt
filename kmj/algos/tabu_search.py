from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any
import numpy as np

from core.base import Objective, State, Algorithm

class BitwiseTabuSearch(Algorithm):
    """
    비트 벡터(0/1) 기반 Tabu Search
    - state: np.ndarray[int] (shape = (num_bits,), 값은 0 또는 1)
    - neighbor 연산:
        * flip: 임의의 index 하나를 0<->1 토글
        * swap: 한 개의 1을 0으로, 한 개의 0을 1로 (1↔0 교환)
    - move 표현:
        * ('flip', idx)
        * ('swap', idx_off, idx_on)  # idx_off: 1->0, idx_on: 0->1
    """

    def __init__(
        self,
        objective: Objective,
        rng: np.random.Generator | None = None,
        max_bs: int = 25,
        tenure: int = 10,
        n_candidates: int = 30,
        max_no_improve: int = 100,
        p_flip: float = 0.5
    ):
        """
        Args:
            num_bits: 비트 수 (즉, state 길이)
            evaluate_fn: cost = evaluate_fn(state)
            rng: numpy random generator (없으면 default_rng 사용)
        """
        super().__init__(objective, rng)
        self.max_bs = max_bs
        self.tenure = tenure
        self.n_candidates = n_candidates
        self.max_no_improve = max_no_improve
        self.p_flip = p_flip

    def _ensure_candidates(self, state: State, n_bs: int | None = None) -> "State":
        """
        state의 초기 상태를 기준으로 후보 위치 리스트 초기화
        """
        if state.bs_layer.sum() > 0: # 이미 초기화되어 있음
            return  
        
        if n_bs is None:
            n_bs = self.max_bs
        state.random_state(n_bs, "even")
        return state
    
    def _init_state(self, state: State, n_bs: int | None = None) -> np.ndarray:
        self._ensure_candidates(state= state, n_bs=n_bs)
        if n_bs is None:
            # 완전 랜덤
            state = self.rng.integers(0, 2, size=self.num_bits, dtype=int)
        else:
            # 정확히 n_ones개의 1을 가지는 상태
            state = np.zeros(self.num_bits, dtype=int)
            n_bs = min(n_bs, self.num_bits)
            idx = self.rng.choice(self.num_bits, size=n_bs, replace=False)
            state[idx] = 1
        return state
    
    def _get_neighbor(self, state: np.ndarray) -> State:
        s = state.copy()

        # flip vs swap 선택
        if self.rng.random() < self.p_flip:
            # === flip ===
            idx = self.rng.integers(0, self.num_bits)
            s[idx] = 1 - s[idx]
            move = ("flip", int(idx))
            return s, move
        else:
            # === swap ===
            ones = np.where(s == 1)[0]
            zeros = np.where(s == 0)[0]

            # swap 불가능하면 fallback으로 flip
            if len(ones) == 0 or len(zeros) == 0:
                idx = self.rng.integers(0, self.num_bits)
                s[idx] = 1 - s[idx]
                move = ("flip", int(idx))
                return s, move

            idx_off = int(ones[self.rng.integers(0, len(ones))])
            idx_on = int(zeros[self.rng.integers(0, len(zeros))])

            s[idx_off] = 0
            s[idx_on] = 1

            # swap은 순서 무의미하니 정렬해서 항상 같은 표현으로 저장
            move = ("swap", idx_off, idx_on)
            return s, move

    def run(
        self,
        max_iter: int,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            max_iter: 최대 iteration
            init_state: 초기 상태 (없으면 랜덤 초기화)
            verbose: 중간 로그 출력 여부
        Returns:
            best_state, best_cost, history (list[dict])
        """
        p = self.params
        rng = self.rng

        # 초기 상태 설정
        if init_state is None:
            current_state = self._init_state(self.max_bs)
        else:
            current_state = np.asarray(init_state, dtype=int)
            if current_state.shape[0] != self.num_bits:
                raise ValueError("init_state length does not match num_bits")

        current_cost = float(self.evaluate_fn(current_state))
        best_state = current_state.copy()
        best_cost = current_cost

        # Tabu 리스트: move(튜플) -> 남은 tenure
        tabu_dict: Dict[Tuple[Any, ...], int] = {}

        history: List[dict] = []
        no_improve_count = 0

        if verbose:
            print(f"[TabuBit] start: cost={current_cost:.4f}")

        for it in range(max_iter):
            candidate_states: List[np.ndarray] = []
            candidate_moves: List[Tuple[Any, ...]] = []
            candidate_costs: List[float] = []

            # 1) 후보 neighbor 여러 개 생성
            for _ in range(p.n_candidates):
                neighbor_state, move = self._get_neighbor(current_state)
                cost = float(self.evaluate_fn(neighbor_state))
                candidate_states.append(neighbor_state)
                candidate_moves.append(move)
                candidate_costs.append(cost)

            # 2) Tabu + Aspiration 기준으로 best candidate 선택
            best_candidate_idx = None
            best_candidate_cost = float("inf")

            for idx, (move_i, cost_i) in enumerate(zip(candidate_moves, candidate_costs)):
                in_tabu = (move_i in tabu_dict and tabu_dict[move_i] > 0)

                # Aspiration: Tabu라도 global best보다 좋으면 허용
                if in_tabu and cost_i >= best_cost:
                    continue

                if cost_i < best_candidate_cost:
                    best_candidate_cost = cost_i
                    best_candidate_idx = idx

            # 만약 모두 Tabu여서 선택이 안 됐으면 그냥 최소 cost 선택
            if best_candidate_idx is None:
                best_candidate_idx = int(np.argmin(candidate_costs))
                best_candidate_cost = candidate_costs[best_candidate_idx]

            best_candidate_state = candidate_states[best_candidate_idx]
            best_candidate_move = candidate_moves[best_candidate_idx]

            # 3) 현재 상태 업데이트
            current_state = best_candidate_state
            current_cost = best_candidate_cost

            # 4) Tabu 리스트 갱신 (선택된 move를 Tabu에 등록)
            if best_candidate_move is not None:
                tabu_dict[best_candidate_move] = p.tenure

            # 5) 글로벌 베스트 갱신
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 6) Tabu tenure 감소 및 만료 제거
            expired = []
            for m in tabu_dict:
                tabu_dict[m] -= 1
                if tabu_dict[m] <= 0:
                    expired.append(m)
            for m in expired:
                del tabu_dict[m]

            # 7) 기록
            history.append(
                {
                    "iter": it,
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "tabu_size": len(tabu_dict),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[TabuBit] iter={it:5d}, "
                    f"current_cost={current_cost:.4f}, best_cost={best_cost:.4f}, "
                    f"tabu_size={len(tabu_dict)}, no_improve={no_improve_count}"
                )

            # 8) 종료 조건
            if no_improve_count >= p.max_no_improve:
                if verbose:
                    print(f"[TabuBit] Early stop: no improvement for {p.max_no_improve} iterations")
                break

        return best_state, best_cost, history


if __name__ == "__main__":
    # 간단 실행 예시 (비트벡터 기반)
    rng = np.random.default_rng(42)
    from objectives import CoverageObjective
    from base import State

    # 50x50 랜덤 트래픽/마스크
    init_state = State.from_shape(50, 50, rng=rng, traffic_params={"pattern": "multi_hotspot"})
    # 목표 함수 (커버리지 전용)
    objective = CoverageObjective(coverage_radius=4.0)

    solver = BitwiseTabuSearch(objective=objective, rng=rng, max_bs=25, tenure=10, n_candidates=30, max_no_improve=50)

    best_state, best_cost, history = solver.run(max_iter=300, init_state=None, verbose=True)

    print("\n=== TabuSearch Result ===")
    print(f"Best cost: {best_cost:.4f}")
    # history 길이 확인
    print(f"History length: {len(history)}")
