import os 
import numpy as np
import matplotlib.pyplot as plt

from core.grid import generate_synthetic_traffic
from core.model import BaseStationProblem, CostWeights
from algos.random_walk import RandomWalkSolver
from algos.simulated_annealing import SimulatedAnnealingSolver
from algos.tabu_search import TabuSearchSolver, TabuParams
from algos.genetic import GeneticAlgorithmSolver, GAParams

def ensure_outdir(path: str = "outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def summarize_results(results, labels):
    print("\n=== 알고리즘 요약 (한 눈에 보기) ===")
    header = f"{'Algorithm':<22}{'Best Cost':>12}{'Best Cov.':>12}"
    print(header)
    print("-" * len(header))
    for label, res in zip(labels, results):
        cov = res.best_metrics.get("coverage_ratio", float('nan'))
        print(f"{label:<22}{res.best_cost:>12.4f}{cov:>12.4f}")

def plot_traffic_and_bs(traffic_map, state, title: str = ""):
    outdir = ensure_outdir()

    H, W = traffic_map.shape
    plt.figure(figsize=(5, 5))
    plt.imshow(traffic_map, origin="lower")
    if state is not None and len(state) > 0:
        xs = state[:, 0]
        ys = state[:, 1]
        plt.scatter(xs, ys, marker="x", color='orange')
    plt.title(title)
    plt.colorbar(label="Traffic")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, title), dpi=200, bbox_inches="tight")
    plt.close()

def plot_all_traffic_and_bs(traffic_map, layouts_dict, filename="all_algos_layouts"):
    """
    layouts_dict: {이름(str): state(np.ndarray 또는 None)} 형태
      예) {
        "Initial": init_state,
        "Random Walk": rw_res.best_state,
        "Simulated Annealing": sa_res.best_state,
        "Tabu Search": tabu_res.best_state,
        "Genetic Algorithm": ga_res.best_state,
      }
    """
    outdir = ensure_outdir()

    H, W = traffic_map.shape
    n = len(layouts_dict)
    names = list(layouts_dict.keys())
    states = list(layouts_dict.values())

    # 2 x 3 그리드 (5개 + 여유 1칸)
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))

    # axes를 1차원 리스트로 편하게 사용
    axes = axes.ravel()

    # 공통으로 쓸 traffic heatmap (첫 번째 서브플롯에서 만든 im을 colorbar에 사용)
    im = None
    for idx, (name, state) in enumerate(zip(names, states)):
        ax = axes[idx]
        im = ax.imshow(traffic_map, origin="lower")
        if state is not None and len(state) > 0:
            xs = state[:, 0]
            ys = state[:, 1]
            ax.scatter(xs, ys, marker="x", color="orange")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    # 남는 서브플롯은 숨기기
    for j in range(len(names), len(axes)):
        axes[j].axis("off")

    # 공통 컬러바
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), label="Traffic", shrink=0.8)

    plt.savefig(os.path.join(outdir, filename), dpi=200, bbox_inches="tight")
    plt.close()
    
def plot_history_best(histories, labels, key="best_cost"):
    outdir = ensure_outdir()

    plt.figure(figsize=(6, 4))
    for hist, label in zip(histories, labels):
        iters = [h["iter"] for h in hist]
        target = [h[key] for h in hist]
        plt.plot(iters, target, label=label)
    plt.xlabel("Iteration / Generation")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"all_algos_{key}"), dpi=200, bbox_inches="tight")
    plt.close()

def plot_chart_best(results, labels, key="best_cost"):
    outdir = ensure_outdir()

    best_target = [
        res.best_metrics.get(key, 0.0)
        for res in results
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, best_target)
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel(key)
    plt.title(f"{key} by Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_best_coverage"), dpi=200, bbox_inches="tight")
    plt.close()

def main():
    rng = np.random.default_rng(42)

    width, height = 50, 50
    traffic_map = generate_synthetic_traffic(
        width=width,
        height=height,
        pattern="multi_hotspot",
        rng=rng,
    )

    n_bs = 30
    coverage_radius = 4.0
    cost_weights = CostWeights(w_uncovered=1.0, w_num_bs=0.0)

    problem = BaseStationProblem(
        traffic_map=traffic_map,
        n_bs=n_bs,
        coverage_radius=coverage_radius,
        cost_weights=cost_weights,
        rng=rng,
    )

    init_state = problem.random_state()

    # 알고리즘 정의
    rw_solver = RandomWalkSolver(problem, rng=rng)
    sa_solver = SimulatedAnnealingSolver(
        problem, rng=rng, T_init=1.0, alpha=0.99, T_min=1e-3
    )
    tabu_solver = TabuSearchSolver(
        problem,
        rng=rng,
        params=TabuParams(tenure=10, n_candidates=30, max_no_improve=100),
    )
    ga_solver = GeneticAlgorithmSolver(
        problem,
        rng=rng,
        params=GAParams(
            pop_size=40,
            n_generations=500,   # GA는 세대 수 기준
            crossover_rate=0.9,
            mutation_rate=0.2,
            tournament_size=3,
            elitism=2,
        ),
    )

    max_iter = 500  # RW / SA / Tabu 에 사용

    print("=== Random Walk ===")
    rw_res = rw_solver.run(max_iter=max_iter, init_state=init_state, verbose=True)
    print("RW best cost:", rw_res.best_cost, "best cov:", rw_res.best_metrics["coverage_ratio"])

    print("\n=== Simulated Annealing ===")
    sa_res = sa_solver.run(max_iter=max_iter, init_state=init_state, verbose=True)
    print("SA best cost:", sa_res.best_cost, "best cov:", sa_res.best_metrics["coverage_ratio"])

    print("\n=== Tabu Search ===")
    tabu_res = tabu_solver.run(max_iter=max_iter, init_state=init_state, verbose=True)
    print("Tabu best cost:", tabu_res.best_cost, "best cov:", tabu_res.best_metrics["coverage_ratio"])

    print("\n=== Genetic Algorithm ===")
    ga_res = ga_solver.run(max_iter=None, init_state=init_state, verbose=True)
    print("GA best cost:", ga_res.best_cost, "best cov:", ga_res.best_metrics["coverage_ratio"])

    # 초기 & 각 알고리즘의 최적 배치 시각화
    # plot_traffic_and_bs(traffic_map, init_state, "Initial Layout")
    # plot_traffic_and_bs(traffic_map, rw_res.best_state, "Random Walk Best")
    # plot_traffic_and_bs(traffic_map, sa_res.best_state, "Simulated Annealing Best")
    # plot_traffic_and_bs(traffic_map, tabu_res.best_state, "Tabu Search Best")
    # plot_traffic_and_bs(traffic_map, ga_res.best_state, "Genetic Algorithm Best")

    layouts_dict = {
        "Initial": init_state,
        "Random Walk": rw_res.best_state,
        "Simulated Annealing": sa_res.best_state,
        "Tabu Search": tabu_res.best_state,
        "Genetic Algorithm": ga_res.best_state,
    }
    plot_all_traffic_and_bs(traffic_map, layouts_dict)

    # 커버리지 수렴 비교
    plot_history_best(
        [rw_res.history, sa_res.history, tabu_res.history, ga_res.history],
        ["Random Walk", "SA", "Tabu", "GA"],
        key="best_cost",
    )

    labels = ["Random Walk", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    results = [rw_res, sa_res, tabu_res, ga_res]
    summarize_results(results, labels)

    plt.show()


if __name__ == "__main__":
    main()
