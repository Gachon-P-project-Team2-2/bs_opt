# algos/simulated_annealing.py
from __future__ import annotations
from typing import List
import numpy as np

from core.base import Algorithm, Objective, State, History


def _sample_neighbor(
    state: State,
    rng: np.random.Generator,
    p_move: float = 0.3,
    p_add: float = 0.3,
    p_remove: float = 0.3,
    min_bs: int = 0,
    max_bs: int | None = None,
) -> State:
    """
    Generate a neighboring State without relying on State.get_neighbor.
    Mirrors the add/move/remove logic from the base module.
    """
    probs = np.array([p_move, p_add, p_remove], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else np.array([1.0, 0.0, 0.0])

    coords = state.get_coordinates()
    n_bs = len(coords)
    max_bs_allowed = max_bs if max_bs is not None else int(np.count_nonzero(state.mask_layer == 0))

    installable = np.argwhere(state.mask_layer == 0)
    installable_set = {(int(x), int(y)) for y, x in installable}
    occupied = set(coords)
    empty_installable = list(installable_set - occupied)

    op = rng.choice(["move", "add", "remove"], p=probs)

    if op == "add" and n_bs < max_bs_allowed and empty_installable:
        y, x = empty_installable[rng.integers(0, len(empty_installable))]
        return state.add_base_station(x, y)

    if op == "remove" and n_bs > min_bs and n_bs > 0:
        idx = rng.integers(0, n_bs)
        x, y = coords[idx]
        return state.remove_base_station(x, y)

    if not coords:
        return state.copy()

    idx = rng.integers(0, len(coords))
    x, y = coords[idx]
    dx = rng.integers(-1, 2)
    dy = rng.integers(-1, 2)
    nx = x + dx
    ny = y + dy
    if not state.is_installable(nx, ny) or state.bs_layer[ny, nx] == 1:
        return state.copy()
    return state.move_base_station((x, y), (nx, ny))


class SimulatedAnnealing(Algorithm):
    def __init__(
        self,
        objective: Objective,
        rng: np.random.Generator | None = None,
        T_init: float = 1.0,
        alpha: float = 0.99,
        T_min: float = 1e-3,
    ):
        super().__init__(objective, rng)
        self.T_init = T_init
        self.alpha = alpha
        self.T_min = T_min

    def run(
        self,
        max_iter: int,
        init_state: State,
        verbose: bool = False,
        neighbor_kwargs: dict | None = None,
    ) -> History:
        if init_state is None:
            raise ValueError("init_state must be provided for simulated annealing")

        neighbor_kwargs = neighbor_kwargs or {}
        objective = self.objective
        current_state = (
            init_state.get_state()
            if hasattr(init_state, "get_state")
            else init_state.copy()
        )

        current_cost, current_metrics = objective.evaluate(current_state)

        best_state = current_state
        best_cost = current_cost
        best_metrics = current_metrics

        history = History()
        T = self.T_init

        for it in range(max_iter):
            neighbor = _sample_neighbor(current_state, self.rng, **neighbor_kwargs)
            neighbor_cost, neighbor_metrics = objective.evaluate(neighbor)

            delta = neighbor_cost - current_cost
            accept = False
            if delta < 0:
                accept = True
            elif T > 1e-12:
                prob = np.exp(-delta / T)
                if self.rng.random() < prob:
                    accept = True

            if accept:
                current_state = neighbor
                current_cost = neighbor_cost
                current_metrics = neighbor_metrics

            if current_cost < best_cost:
                best_state = current_state
                best_cost = current_cost
                best_metrics = current_metrics

            history.add_record(
                {
                    "iter": it,
                    "T": float(T),
                    "current_cost": float(current_cost),
                    "best_cost": float(best_cost),
                    "current_coverage": float(current_metrics.get("coverage_ratio", np.nan)),
                    "best_coverage": float(best_metrics.get("coverage_ratio", np.nan)),
                }
            )

            if verbose and (it % max(1, max_iter // 10) == 0):
                print(
                    f"[SA] iter={it:5d}, T={T:.4f}, current_cost={current_cost:.4f}, "
                    f"best_cost={best_cost:.4f}, best_cov={best_metrics.get('coverage_ratio', float('nan')):.4f}"
                )

            T = max(self.T_min, T * self.alpha)

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
    import numpy as np
    import matplotlib.pyplot as plt

    # =========================
    # 기본 세팅
    # =========================
    rng = np.random.default_rng(42)

    # 초기 State 생성
    # (random_walk, genetic과 동일한 방식)
    init_state = State.from_shape(
        50, 50,
        rng=rng,
        traffic_params={"pattern": "multi_hotspot"},
    )

    # Objective 생성
    from objectives import CoverageObjective
    objective = CoverageObjective(k_scale=1e8, beta=4.5)

    # =========================
    # Simulated Annealing 실행
    # =========================
    sa = SimulatedAnnealing(
        objective=objective,
        rng=rng,
        T_init=1.0,
        alpha=0.995,
        T_min=1e-4,
    )

    history = sa.run(
        max_iter=10000,
        init_state=init_state,
        verbose=True,
        neighbor_kwargs={},   # State.get_neighbor(**kwargs)
    )

    # =========================
    # 결과 정리
    # =========================
    last = history.get_history()[-1]
    best_state: State = last["best_state"]
    best_cost = last["best_cost"]
    best_metrics = last["best_metrics"]

    print("\n=== Simulated Annealing Result ===")
    print(f"Best cost: {best_cost:.6f}")
    print(f"Best coverage_ratio: {best_metrics.get('coverage_ratio', float('nan')):.6f}")
    print(f"Best coverage_percent: {best_metrics.get('coverage_percent', float('nan')):.4f}")
    print(f"Num BS: {len(best_state.get_coordinates())}")

    # =========================
    # 결과 시각화 (파일 저장)
    # =========================
    def plot_layout_save(state: State, traffic: np.ndarray, title: str, path: str):
        plt.figure(figsize=(6, 6))
        plt.imshow(traffic, origin="lower")
        coords = state.get_coordinates()
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            plt.scatter(xs, ys, marker="x", color="orange")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    plot_layout_save(
        init_state,
        init_state.traffic_layer,
        "Initial layout (SA)",
        "sa_initial_layout.png",
    )

    plot_layout_save(
        best_state,
        init_state.traffic_layer,
        "Best layout (SA)",
        "sa_best_layout.png",
    )

    # 비용 히스토리
    hist = history.get_history()
    iters = [h["iter"] for h in hist if "iter" in h and "best_cost" in h]
    best_costs = [h["best_cost"] for h in hist if "iter" in h and "best_cost" in h]

    plt.figure(figsize=(7, 4))
    plt.plot(iters, best_costs, label="Best cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Simulated Annealing Best Cost History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sa_cost_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" - sa_initial_layout.png")
    print(" - sa_best_layout.png")
    print(" - sa_cost_history.png")
