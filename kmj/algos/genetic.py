# algos/genetic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from core.base import Algorithm, History, Objective, State


@dataclass
class GAParams:
    pop_size: int = 40            # population 크기
    n_generations: int = 200      # 세대 수 (max_iter 대신 사용)
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism: int = 2              # 상위 몇 개를 그대로 다음 세대로 복사

@dataclass
class VarGAParams:
    pop_size: int = 40
    n_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism: int = 2
    min_bs: int = 1
    max_bs: int = 50
    p_add: float = 0.2
    p_remove: float = 0.2
    p_move: float = 0.6
    

class Genetic(Algorithm):
    """
    Objective 기반 GA.
    - n_bs가 주어지면 고정 길이 GA
    - VarGAParams가 주어지면 가변 길이 GA (add/remove/move)
    """

    def __init__(
        self,
        objective: Objective,
        base_state: State,
        n_bs: int | None = None,
        rng: np.random.Generator | None = None,
        params: GAParams | VarGAParams | None = None,
    ):
        super().__init__(objective, rng)
        self.base_state = base_state
        self.height = base_state.height
        self.width = base_state.width

        # var-length 여부 결정
        self.is_var = isinstance(params, VarGAParams) or n_bs is None
        self.params = params or (VarGAParams() if self.is_var else GAParams())
        if not self.is_var and n_bs is None:
            raise ValueError("n_bs must be provided for fixed-length GA")
        self.n_bs = n_bs if n_bs is not None else 0  # var일 때는 _random_individual에서 결정

    def _random_individual(self) -> np.ndarray:
        if self.is_var:
            k = int(self.rng.integers(self.params.min_bs, self.params.max_bs + 1))
            xs = self.rng.integers(0, self.width, size=k)
            ys = self.rng.integers(0, self.height, size=k)
        else:
            xs = self.rng.integers(0, self.width, size=self.n_bs)
            ys = self.rng.integers(0, self.height, size=self.n_bs)
        return np.stack([xs, ys], axis=1).astype(int)

    def _coords_to_state(self, coords: np.ndarray) -> State:
        coords = np.asarray(coords, dtype=int)
        bs_layer = np.zeros_like(self.base_state.bs_layer, dtype=int)
        for x, y in coords:
            if self.base_state.is_installable(int(x), int(y)):
                bs_layer[int(y), int(x)] = 1
        return State(
            traffic_layer=self.base_state.traffic_layer,
            mask_layer=self.base_state.mask_layer,
            bs_layer=bs_layer,
            rng=self.rng,
        )

    def _evaluate_population(
        self, population: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[dict]]:
        costs = []
        metrics_list = []
        for ind in population:
            c, m = self.objective.evaluate(self._coords_to_state(ind))
            costs.append(c)
            metrics_list.append(m)
        return np.array(costs, dtype=float), metrics_list

    def _tournament_select(
        self, population: List[np.ndarray], costs: np.ndarray
    ) -> np.ndarray:
        idxs = self.rng.integers(0, len(population), size=self.params.tournament_size)
        best_idx = idxs[0]
        best_cost = costs[best_idx]
        for i in idxs[1:]:
            if costs[i] < best_cost:
                best_idx = i
                best_cost = costs[i]
        return population[best_idx].copy()

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.params.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.is_var:
            len1, len2 = parent1.shape[0], parent2.shape[0]
            if len1 == 0 or len2 == 0:
                return parent1.copy(), parent2.copy()
            point = self.rng.integers(1, min(len1, len2) + 1)
            child1 = np.vstack([parent1[:point], parent2[point:]])
            child2 = np.vstack([parent2[:point], parent1[point:]])
            child1 = child1[: self.params.max_bs]
            child2 = child2[: self.params.max_bs]
        else:
            p1 = parent1.reshape(-1)
            p2 = parent2.reshape(-1)
            length = p1.shape[0]
            if length <= 1:
                return parent1.copy(), parent2.copy()
            point = self.rng.integers(1, length)
            child1 = np.concatenate([p1[:point], p2[point:]]).reshape(parent1.shape)
            child2 = np.concatenate([p2[:point], p1[point:]]).reshape(parent2.shape)

        child1 = child1.copy()
        child2 = child2.copy()
        child1[:, 0] = np.clip(child1[:, 0], 0, self.width - 1)
        child1[:, 1] = np.clip(child1[:, 1], 0, self.height - 1)
        child2[:, 0] = np.clip(child2[:, 0], 0, self.width - 1)
        child2[:, 1] = np.clip(child2[:, 1], 0, self.height - 1)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        ind = individual.copy()
        if self.rng.random() > self.params.mutation_rate:
            return ind

        if self.is_var:
            ops = ["add", "remove", "move"]
            probs = [self.params.p_add, self.params.p_remove, self.params.p_move]
            op = self.rng.choice(ops, p=np.array(probs) / np.sum(probs))
            if op == "add" and ind.shape[0] < self.params.max_bs:
                new_x = self.rng.integers(0, self.width)
                new_y = self.rng.integers(0, self.height)
                ind = np.vstack([ind, np.array([[new_x, new_y]], dtype=int)])
            elif op == "remove" and ind.shape[0] > self.params.min_bs:
                idx = self.rng.integers(0, ind.shape[0])
                ind = np.delete(ind, idx, axis=0)
            else:  # move
                if ind.shape[0] == 0:
                    return ind
                idx = self.rng.integers(0, ind.shape[0])
                dx = self.rng.integers(-1, 2)
                dy = self.rng.integers(-1, 2)
                ind[idx, 0] = np.clip(ind[idx, 0] + dx, 0, self.width - 1)
                ind[idx, 1] = np.clip(ind[idx, 1] + dy, 0, self.height - 1)
        else:
            idx = self.rng.integers(0, ind.shape[0])
            dx = self.rng.integers(-1, 2)
            dy = self.rng.integers(-1, 2)
            ind[idx, 0] = np.clip(ind[idx, 0] + dx, 0, self.width - 1)
            ind[idx, 1] = np.clip(ind[idx, 1] + dy, 0, self.height - 1)
        return ind

    def run(
        self,
        max_iter: int | None = None,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> History:
        n_gen = self.params.n_generations

        population: List[np.ndarray] = []
        if init_state is not None:
            population.append(np.asarray(init_state, dtype=int))
            for _ in range(self.params.pop_size - 1):
                population.append(self._random_individual())
        else:
            for _ in range(self.params.pop_size):
                population.append(self._random_individual())

        costs, metrics_list = self._evaluate_population(population)
        best_idx = int(np.argmin(costs))
        best_state = population[best_idx].copy()
        best_cost = float(costs[best_idx])
        best_metrics = metrics_list[best_idx].copy()

        history = History()

        for gen in range(n_gen):
            new_population: List[np.ndarray] = []

            elite_indices = np.argsort(costs)[: self.params.elitism]
            for idx in elite_indices:
                new_population.append(population[int(idx)].copy())

            while len(new_population) < self.params.pop_size:
                parent1 = self._tournament_select(population, costs)
                parent2 = self._tournament_select(population, costs)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.params.pop_size:
                    new_population.append(child2)

            population = new_population
            costs, metrics_list = self._evaluate_population(population)

            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            gen_best_state = population[gen_best_idx].copy()
            gen_best_metrics = metrics_list[gen_best_idx].copy()

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_state = gen_best_state.copy()
                best_metrics = gen_best_metrics.copy()

            history.add_record(
                {
                    "iter": gen,
                    "current_best_cost": float(gen_best_cost),
                    "best_cost": float(best_cost),
                    "current_best_coverage": float(gen_best_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                    "best_state": best_state.copy(),
                }
            )

            if verbose and (gen % max(1, n_gen // 10) == 0):
                print(
                    f"[GA-Obj] gen={gen:4d}, gen_best_cost={gen_best_cost:.4f}, "
                    f"global_best_cost={best_cost:.4f}, "
                    f"global_best_cov={best_metrics['coverage_ratio']:.4f}"
                )

        history.add_record(
            {
                "iter": n_gen,
                "best_cost": float(best_cost),
                "best_metrics": best_metrics,
                "best_state": best_state.copy(),
            }
        )
        return history

class VarGenetic(Algorithm):
    """
    기지국 수가 가변인 GA. add/remove/move를 포함한 변이를 통해
    커버리지 최대화와 기지국 수 최소화를 동시에 탐색한다.
    """

    def __init__(
        self,
        objective: Objective,
        base_state: State,
        rng: np.random.Generator | None = None,
        params: VarGAParams | None = None,
    ):
        super().__init__(objective, rng)
        self.params = params or VarGAParams()
        self.base_state = base_state
        self.height = base_state.height
        self.width = base_state.width

    # population 관련 유틸
    def _random_individual(self) -> np.ndarray:
        k = int(self.rng.integers(self.params.min_bs, self.params.max_bs + 1))
        xs = self.rng.integers(0, self.width, size=k)
        ys = self.rng.integers(0, self.height, size=k)
        return np.stack([xs, ys], axis=1).astype(int)

    def _coords_to_state(self, coords: np.ndarray) -> State:
        coords = np.asarray(coords, dtype=int)
        bs_layer = np.zeros_like(self.base_state.bs_layer, dtype=int)
        for x, y in coords:
            if self.base_state.is_installable(int(x), int(y)):
                bs_layer[int(y), int(x)] = 1
        return State(
            traffic_layer=self.base_state.traffic_layer,
            mask_layer=self.base_state.mask_layer,
            bs_layer=bs_layer,
            rng=self.rng,
        )

    def _evaluate_population(self, population: List[np.ndarray]) -> Tuple[np.ndarray, List[dict]]:
        costs = []
        metrics_list = []
        for ind in population:
            c, m = self.objective.evaluate(self._coords_to_state(ind))
            costs.append(c)
            metrics_list.append(m)
        return np.array(costs, dtype=float), metrics_list

    def _tournament_select(self, population: List[np.ndarray], costs: np.ndarray) -> np.ndarray:
        idxs = self.rng.integers(0, len(population), size=self.params.tournament_size)
        best_idx = idxs[0]
        best_cost = costs[best_idx]
        for i in idxs[1:]:
            if costs[i] < best_cost:
                best_idx = i
                best_cost = costs[i]
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.params.crossover_rate:
            return parent1.copy(), parent2.copy()
        len1, len2 = parent1.shape[0], parent2.shape[0]
        if len1 == 0 or len2 == 0:
            return parent1.copy(), parent2.copy()
        point = self.rng.integers(1, min(len1, len2) + 1)
        child1 = np.vstack([parent1[:point], parent2[point:]])
        child2 = np.vstack([parent2[:point], parent1[point:]])
        # 길이 클램프
        child1 = child1[: self.params.max_bs]
        child2 = child2[: self.params.max_bs]
        child1[:, 0] = np.clip(child1[:, 0], 0, self.width - 1)
        child1[:, 1] = np.clip(child1[:, 1], 0, self.height - 1)
        child2[:, 0] = np.clip(child2[:, 0], 0, self.width - 1)
        child2[:, 1] = np.clip(child2[:, 1], 0, self.height - 1)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        ind = individual.copy()
        if self.rng.random() > self.params.mutation_rate:
            return ind

        ops = ["add", "remove", "move"]
        probs = [self.params.p_add, self.params.p_remove, self.params.p_move]
        op = self.rng.choice(ops, p=np.array(probs) / np.sum(probs))

        if op == "add" and ind.shape[0] < self.params.max_bs:
            new_x = self.rng.integers(0, self.width)
            new_y = self.rng.integers(0, self.height)
            ind = np.vstack([ind, np.array([[new_x, new_y]], dtype=int)])
        elif op == "remove" and ind.shape[0] > self.params.min_bs:
            idx = self.rng.integers(0, ind.shape[0])
            ind = np.delete(ind, idx, axis=0)
        else:  # move
            if ind.shape[0] == 0:
                return ind
            idx = self.rng.integers(0, ind.shape[0])
            dx = self.rng.integers(-1, 2)
            dy = self.rng.integers(-1, 2)
            ind[idx, 0] = np.clip(ind[idx, 0] + dx, 0, self.width - 1)
            ind[idx, 1] = np.clip(ind[idx, 1] + dy, 0, self.height - 1)

        return ind

    def run(
        self,
        max_iter: int | None = None,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> History:
        n_gen = self.params.n_generations

        population: List[np.ndarray] = []
        if init_state is not None:
            population.append(np.asarray(init_state, dtype=int))
            for _ in range(self.params.pop_size - 1):
                population.append(self._random_individual())
        else:
            for _ in range(self.params.pop_size):
                population.append(self._random_individual())

        costs, metrics_list = self._evaluate_population(population)
        best_idx = int(np.argmin(costs))
        best_state = population[best_idx].copy()
        best_cost = float(costs[best_idx])
        best_metrics = metrics_list[best_idx].copy()

        history = History()

        for gen in range(n_gen):
            new_population: List[np.ndarray] = []

            elite_indices = np.argsort(costs)[: self.params.elitism]
            for idx in elite_indices:
                new_population.append(population[int(idx)].copy())

            while len(new_population) < self.params.pop_size:
                parent1 = self._tournament_select(population, costs)
                parent2 = self._tournament_select(population, costs)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.params.pop_size:
                    new_population.append(child2)

            population = new_population
            costs, metrics_list = self._evaluate_population(population)

            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            gen_best_state = population[gen_best_idx].copy()
            gen_best_metrics = metrics_list[gen_best_idx].copy()

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_state = gen_best_state.copy()
                best_metrics = gen_best_metrics.copy()

            history.add_record(
                {
                    "iter": gen,
                    "current_best_cost": float(gen_best_cost),
                    "best_cost": float(best_cost),
                    "current_best_coverage": float(gen_best_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                    "num_bs": int(best_state.shape[0]),
                    "best_state": best_state.copy(),
                }
            )

            if verbose and (gen % max(1, n_gen // 10) == 0):
                print(
                    f"[GA-Var] gen={gen:4d}, gen_best_cost={gen_best_cost:.4f}, "
                    f"global_best_cost={best_cost:.4f}, "
                    f"global_best_cov={best_metrics['coverage_ratio']:.4f}, "
                    f"num_bs={best_state.shape[0]}"
                )

        history.add_record(
            {
                "iter": n_gen,
                "best_cost": float(best_cost),
                "best_metrics": best_metrics,
                "best_state": best_state.copy(),
                "num_bs": int(best_state.shape[0]),
            }
        )
        return history

if __name__ == "__main__":
    import numpy as np

    # 프로젝트 구조에 맞게 State / Objective 생성
    # (random_walk.py에서 쓰던 것과 동일한 방식)
    rng = np.random.default_rng(42)

    # base State 생성 (50x50, 트래픽/마스크 포함)
    base_state = State.from_shape(
        50, 50,
        rng=rng,
        traffic_params={"pattern": "multi_hotspot"},
    )

    # Objective 생성
    from objectives import CoverageObjective
    objective = CoverageObjective(k_scale=1e8, beta=4.5)

    # -----------------------------
    # 1) 가변 길이 GA 실행 (추천)
    # -----------------------------
    params = VarGAParams(
        pop_size=40,
        n_generations=10000,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_size=3,
        elitism=2,
        min_bs=1,
        max_bs=50,
        p_add=0.2,
        p_remove=0.2,
        p_move=0.6,
    )

    ga = Genetic(
        objective=objective,
        base_state=base_state,
        n_bs=None,          # var-length로 돌릴 때는 None
        rng=rng,
        params=params,
    )

    hist = ga.run(verbose=True)

    # 최종 기록(마지막 add_record)에 best_state, best_metrics가 들어있음
    last = hist.get_history()[-1]
    best_coords = np.asarray(last["best_state"], dtype=int)

    best_state = ga._coords_to_state(best_coords)
    best_cost, best_metrics = objective.evaluate(best_state)

    print("\n=== Genetic Algorithm Result ===")
    print(f"Best cost: {best_cost:.6f}")
    print(f"Best coverage_ratio: {best_metrics.get('coverage_ratio', float('nan')):.6f}")
    print(f"Best coverage_percent: {best_metrics.get('coverage_percent', float('nan')):.4f}")
    print(f"Num BS: {len(best_state.get_coordinates())}")
    print("Best BS coords (first 10):", best_coords[:10].tolist())

    # -----------------------------
    # (선택) 결과 시각화 저장
    # -----------------------------
    import matplotlib.pyplot as plt

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

    plot_layout_save(base_state, base_state.traffic_layer, "Initial layout (GA)", "ga_initial_layout.png")
    plot_layout_save(best_state, base_state.traffic_layer, "Best layout (GA)", "ga_best_layout.png")

    # 비용 히스토리 저장
    history = hist.get_history()
    iters = [h["iter"] for h in history if "iter" in h and "best_cost" in h]
    best_costs = [h["best_cost"] for h in history if "iter" in h and "best_cost" in h]

    plt.figure(figsize=(7, 4))
    plt.plot(iters, best_costs, label="Best cost")
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA Best Cost History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_cost_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: ga_initial_layout.png, ga_best_layout.png, ga_cost_history.png")
