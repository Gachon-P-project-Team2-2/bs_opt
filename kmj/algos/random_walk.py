# algos/random_walk.py
from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

from core.base import Algorithm, State, History, Objective


class BitwiseRandomWalk(Algorithm):
    """
    RandomWalk (비트열 기반):

    - 설치 가능 후보 위치 리스트(candidate_positions)를 만든 뒤,
      길이 n_candidates인 이진 비트열로 상태를 표현한다.
    - 이웃 생성 연산:
        * 50%: Flip (비트 하나 0<->1 토글)
        * 50%: Swap (1인 비트 하나 끄고, 0인 비트 하나 켜기)
    - 목적: 기지국 수와 커버리지를 논문식 비용 f = K * N / R^β 기준으로 최소화.
    """

    def __init__(
        self,
        objective: Objective,
        p_accept_worse: float = 0.03,
        max_bs: int = 50,
        init_bs: int = 25,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(objective, rng)
        self.p_accept_worse = p_accept_worse
        self.max_bs = max_bs
        self.init_bs = init_bs

    def _init_state(self, state: State) -> tuple[list[tuple[int, int]], np.ndarray]:
        """
        state 기반으로 후보 좌표와 초기 비트열을 초기화한다.
        반환: (candidate_coords, current_bits)
        """
        state = state.random_state(self.max_bs, "uniform")
        candidate_coords = state.get_coordinates()

        init_bs = min(self.init_bs, len(candidate_coords))
        if init_bs < self.init_bs:
            print(f"Warning: init_bs={self.init_bs} -> {init_bs} (후보 위치 수에 맞춤)")

        current_bits = [0] * init_bs + [1] * (len(candidate_coords) - init_bs)
        self.rng.shuffle(current_bits)
        current_bits = np.asarray(current_bits, dtype=int)

        return candidate_coords, current_bits

    def _state_from_bits(
        self, state: State, candidate_coords: list[tuple[int, int]], bits: np.ndarray
    ) -> State:
        """비트열로부터 새로운 State 생성"""
        bs_layer = np.zeros_like(state.bs_layer, dtype=int)
        for i, (x, y) in enumerate(candidate_coords):
            if bits[i] == 1:
                bs_layer[y, x] = 1

        return State(
            traffic_layer=state.traffic_layer,
            mask_layer=state.mask_layer,
            bs_layer=bs_layer,
            rng=state.rng,
        )

    def _get_neighbor(self, bits: np.ndarray) -> np.ndarray:
        """비트열 이웃 생성: 50% flip, 50% swap"""
        neighbor = np.asarray(bits, dtype=int).copy()
        n = len(bits)

        if n <= 0:
            return neighbor

        if self.rng.random() < 0.5:
            # Flip
            idx = self.rng.integers(0, n)
            neighbor[idx] = 1 - neighbor[idx]
        else:
            ones = np.where(neighbor == 1)[0]
            zeros = np.where(neighbor == 0)[0]

            if len(ones) > 0 and len(zeros) > 0:
                idx_off = int(self.rng.choice(ones))
                idx_on = int(self.rng.choice(zeros))
                neighbor[idx_off] = 0
                neighbor[idx_on] = 1
            else:
                # swap 불가일 때는 flip으로 대체
                idx = self.rng.integers(0, n)
                neighbor[idx] = 1 - neighbor[idx]

        return neighbor

    def run(
        self,
        max_iter: int,
        init_state: State,
        verbose: bool = False,
    ) -> History:

        candidate_coords, current_bits = self._init_state(init_state)

        current_state = self._state_from_bits(init_state, candidate_coords, current_bits)
        current_cost, current_metrics = self.objective.evaluate(current_state)

        best_bits = current_bits.copy()
        best_cost = current_cost
        best_metrics = current_metrics

        history = History()
        p = self.p_accept_worse

        for it in range(max_iter):
            neighbor_bits = self._get_neighbor(current_bits)
            neighbor_state = self._state_from_bits(init_state, candidate_coords, neighbor_bits)
            neighbor_cost, neighbor_metrics = self.objective.evaluate(neighbor_state)

            if neighbor_cost <= current_cost:
                current_bits = neighbor_bits
                current_state = neighbor_state
                current_cost = neighbor_cost
                current_metrics = neighbor_metrics
            else:
                if self.rng.random() < p:
                    current_bits = neighbor_bits
                    current_state = neighbor_state
                    current_cost = neighbor_cost
                    current_metrics = neighbor_metrics

            if current_cost < best_cost:
                best_cost = current_cost
                best_bits = current_bits.copy()
                best_metrics = current_metrics

            history.add_record(
                {
                    "iter": it,
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics.get("coverage_ratio", np.nan)),
                    "best_coverage": float(best_metrics.get("coverage_ratio", np.nan)),
                    "num_bs": int(current_bits.sum()),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[RandomWalk] iter={it:5d}, "
                    f"current_cost={current_cost:.4f}, best_cost={best_cost:.4f}, "
                    f"num_bs={int(current_bits.sum())}"
                )

        best_state = self._state_from_bits(init_state, candidate_coords, best_bits)
        history.add_record(
            {
                "iter": max_iter,
                "best_state": best_state,
                "best_cost": float(best_cost),
                "best_metrics": best_metrics,
            }
        )
        return history


class MovableRandomWalk(Algorithm):
    """
    MovableRandomWalk (설치 후보 위치 비고정, 비트열 사용 X)

    핵심:
      - 후보 리스트/비트열 없이, 현재 State의 기지국 좌표와 mask_layer를 직접 보고
        Move / Remove 연산을 수행한다.
      - Move: 임의의 기지국을 다른 설치 가능 위치로 옮김
      - Remove: 임의의 기지국을 제거 (기지국 수 감소)
    """

    def __init__(
        self,
        objective: Objective,
        rng: np.random.Generator | None = None,
        p_move: float = 0.5,
        p_accept_worse: float = 0.03,
        max_bs: int = 25,
    ):
        super().__init__(objective, rng)
        self.p_move = p_move
        self.p_accept_worse = p_accept_worse
        self.max_bs = max_bs

    def _ensure_candidates(self, state: State, n_bs: int | None = None) -> State:
        if state.bs_layer.sum() > 0:
            return state.copy()

        max_installable = int((state.mask_layer == 0).sum())
        if max_installable <= 0:
            raise ValueError("No installable cells available for initialization")

        target_bs = n_bs if n_bs is not None else self.max_bs
        target_bs = min(target_bs, max_installable)
        return state.random_state(target_bs, "even")

    def _get_neighbor(self, state: State) -> State:
        coords = state.get_coordinates()
        n_bs = len(coords)

        if n_bs == 0:
            return state.copy()

        r = self.rng.random()

        if r < self.p_move:
            idx = self.rng.integers(0, n_bs)
            x, y = coords[idx]

            installable = np.argwhere(state.mask_layer == 0)
            occupied = set(coords)
            candidates: List[Tuple[int, int]] = []
            for yy, xx in installable:
                cx, cy = int(xx), int(yy)
                if (cx, cy) not in occupied:
                    candidates.append((cx, cy))

            if not candidates:
                return state.copy()

            nx, ny = candidates[self.rng.integers(0, len(candidates))]
            return state.move_base_station((x, y), (nx, ny))
        else:
            idx = self.rng.integers(0, n_bs)
            x, y = coords[idx]
            return state.remove_base_station(x, y)

    def run(
        self,
        max_iter: int,
        init_state: State,
        verbose: bool = False,
    ) -> History:
        current_state = self._ensure_candidates(init_state)

        current_cost, current_metrics = self.objective.evaluate(current_state)

        best_state = current_state
        best_cost = current_cost
        best_metrics = current_metrics

        history = History()

        for it in range(max_iter):
            neighbor = self._get_neighbor(current_state)
            neighbor_cost, neighbor_metrics = self.objective.evaluate(neighbor)

            if neighbor_cost <= current_cost:
                current_state = neighbor
                current_cost = neighbor_cost
                current_metrics = neighbor_metrics
            else:
                if self.rng.random() < self.p_accept_worse:
                    current_state = neighbor
                    current_cost = neighbor_cost
                    current_metrics = neighbor_metrics

            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state
                best_metrics = neighbor_metrics

            history.add_record(
                {
                    "iter": it,
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics.get("coverage_ratio", np.nan)),
                    "best_coverage": float(best_metrics.get("coverage_ratio", np.nan)),
                    "num_bs": len(current_state.get_coordinates()),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[MovableRandomWalk] iter={it:5d}, "
                    f"current_cost={current_cost:.4f}, best_cost={best_cost:.4f}, "
                    f"num_bs={len(current_state.get_coordinates())}"
                )

        history.add_record(
            {
                "iter": max_iter,
                "best_state": best_state,
                "best_cost": float(best_cost),
                "best_metrics": best_metrics,
            }
        )
        return history


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    init_state = State.from_shape(50, 50, rng=rng, traffic_params={"pattern": "multi_hotspot"})

    from objectives import CoverageObjective
    objective = CoverageObjective(k_scale=1e8, beta=4.5)

    rw = BitwiseRandomWalk(objective=objective, rng=rng, max_bs=50, init_bs=25)
    hist = rw.run(max_iter=10000, init_state=init_state, verbose=True)

    best = hist.get_history()[-1]
    print("\n=== RandomWalk Result ===")
    print(f"Best cost: {best['best_cost']:.4f}")
    print(f"Best coverage: {best['best_metrics'].get('coverage_ratio', float('nan')):.4f}")
    print(f"Num BS: {len(best['best_state'].get_coordinates())}")

    # 기존 시각화 (비용 히스토리 + 레이아웃)도 파일로 저장
    import matplotlib.pyplot as plt

    def plot_layout(state: State, traffic: np.ndarray, title: str):
        plt.figure(figsize=(5, 5))
        plt.imshow(traffic, origin="lower")
        coords = state.get_coordinates()
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            plt.scatter(xs, ys, marker="x", color="orange")
        plt.title(title)
        plt.tight_layout()

    history = hist.get_history()
    iters = [h["iter"] for h in history if "iter" in h and "best_cost" in h]
    best_costs = [h["best_cost"] for h in history if "iter" in h and "best_cost" in h]

    plt.figure(figsize=(6, 4))
    plt.plot(iters, best_costs, label="Best cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("RandomWalk Cost History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("random_walk_cost_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_layout(init_state, init_state.traffic_layer, "Initial layout")
    plt.savefig("initial_layout.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_layout(best["best_state"], init_state.traffic_layer, "Best layout")
    plt.savefig("best_layout.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved base plots: random_walk_cost_history.png, initial_layout.png, best_layout.png")
    print("Saved overlays: single_bs_overlay.png, bs_coverages/*.png")
