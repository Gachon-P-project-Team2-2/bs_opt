# algos/genetic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from algos.base import Solver, SolverResult


@dataclass
class GAParams:
    pop_size: int = 40            # population 크기
    n_generations: int = 200      # 세대 수 (max_iter 대신 사용)
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elitism: int = 2              # 상위 몇 개를 그대로 다음 세대로 복사


class GeneticAlgorithmSolver(Solver):
    """
    간단한 GA 구현
    - 개체: (n_bs, 2) 정수 좌표 배열
    - single-point crossover (flatten 후)
    - 좌표 기반 mutation
    """

    def __init__(
        self,
        problem,
        rng: np.random.Generator | None = None,
        params: GAParams | None = None,
    ):
        super().__init__(problem, rng)
        self.params = params or GAParams()

    # -------------------------------
    # population 관련 유틸
    # -------------------------------
    def _random_individual(self) -> np.ndarray:
        return self.problem.random_state()

    def _evaluate_population(
        self, population: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[dict]]:
        costs = []
        metrics_list = []
        for ind in population:
            c, m = self.problem.evaluate(ind)
            costs.append(c)
            metrics_list.append(m)
        return np.array(costs, dtype=float), metrics_list

    def _tournament_select(
        self,
        population: List[np.ndarray],
        costs: np.ndarray,
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

        # (n_bs, 2) → flatten 후 single-point crossover
        p1 = parent1.reshape(-1)
        p2 = parent2.reshape(-1)
        length = p1.shape[0]

        if length <= 1:
            return parent1.copy(), parent2.copy()

        point = self.rng.integers(1, length)  # [1, length-1] 중 하나

        child1 = np.concatenate([p1[:point], p2[point:]])
        child2 = np.concatenate([p2[:point], p1[point:]])

        child1 = child1.reshape(parent1.shape)
        child2 = child2.reshape(parent2.shape)

        # 범위 밖 좌표 클램프
        child1 = self.problem.clamp_state(child1)
        child2 = self.problem.clamp_state(child2)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        ind = individual.copy()
        if self.rng.random() > self.params.mutation_rate:
            return ind

        n_bs = ind.shape[0]
        idx = self.rng.integers(0, n_bs)

        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)

        ind[idx, 0] = np.clip(ind[idx, 0] + dx, 0, self.problem.width - 1)
        ind[idx, 1] = np.clip(ind[idx, 1] + dy, 0, self.problem.height - 1)

        return ind

    # -------------------------------
    # main run
    # -------------------------------
    def run(
        self,
        max_iter: int | None = None,
        init_state: np.ndarray | None = None,
        verbose: bool = False,
    ) -> SolverResult:
        # max_iter 인자를 무시하고 self.params.n_generations 사용
        n_gen = self.params.n_generations

        # 초기 population 생성
        population: List[np.ndarray] = []
        if init_state is not None:
            # init_state 를 한 개체로 포함시키고 나머지를 랜덤으로
            population.append(self.problem.clamp_state(init_state))
            for _ in range(self.params.pop_size - 1):
                population.append(self._random_individual())
        else:
            for _ in range(self.params.pop_size):
                population.append(self._random_individual())

        costs, metrics_list = self._evaluate_population(population)

        # 초기 best
        best_idx = int(np.argmin(costs))
        best_state = population[best_idx].copy()
        best_cost = float(costs[best_idx])
        best_metrics = metrics_list[best_idx].copy()

        history: List[dict] = []

        for gen in range(n_gen):
            new_population: List[np.ndarray] = []

            # Elitism: 상위 elitism 개체는 그대로 복사
            elite_indices = np.argsort(costs)[: self.params.elitism]
            for idx in elite_indices:
                new_population.append(population[int(idx)].copy())

            # 나머지 개체는 selection + crossover + mutation으로 생성
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

            # 글로벌 best 갱신
            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_state = gen_best_state.copy()
                best_metrics = gen_best_metrics.copy()

            history.append(
                {
                    "iter": gen,
                    "current_best_cost": float(gen_best_cost),
                    "best_cost": float(best_cost),
                    "current_best_coverage": float(gen_best_metrics["coverage_ratio"]),
                    "best_coverage": float(best_metrics["coverage_ratio"]),
                }
            )

            if verbose and (gen % max(1, n_gen // 10) == 0):
                print(
                    f"[GA] gen={gen:4d}, gen_best_cost={gen_best_cost:.4f}, "
                    f"global_best_cost={best_cost:.4f}, "
                    f"global_best_cov={best_metrics['coverage_ratio']:.4f}"
                )

        return SolverResult(
            best_state=best_state,
            best_cost=float(best_cost),
            best_metrics=best_metrics,
            history=history,
        )
