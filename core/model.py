# core/model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CostWeights:
    """비용 함수 가중치"""
    w_uncovered: float = 1.0  # 커버되지 않은 트래픽 패널티
    w_num_bs: float = 0.0     # 기지국 개수 패널티 (추후 사용 가능)


class BaseStationProblem:
    """
    기지국 설치 최적화 문제 정의
    - 상황
      - 격자 기반 서비스 지역
      - traffic_map: (k, 2) shape의 np.ndarray, 각 행이 (x, y) 좌표 (열, 행)
    - 목적: 기지국 설치 수가 주어졌을 때 커버리지가 최대가 되도록 기지국 설치 위치를 결정하라
    """

    def __init__(
        self,
        traffic_map: np.ndarray,
        n_bs: int,
        coverage_radius: float,
        cost_weights: CostWeights | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        traffic_map: (H, W) 2D array
        n_bs: 설치할 기지국 개수 (고정)
        coverage_radius: 커버리지 반경 (격자 단위, Euclidean 거리)
        """
        self.traffic_map = traffic_map.astype(float)
        self.height, self.width = traffic_map.shape
        self.n_bs = n_bs
        self.coverage_radius = float(coverage_radius)
        self.r2 = self.coverage_radius**2

        self.cost_weights = cost_weights or CostWeights()
        self.rng = rng or np.random.default_rng()

        # grid 좌표 precompute
        y, x = np.mgrid[0:self.height, 0:self.width]
        self.grid_x = x  # shape(H, W)
        self.grid_y = y  # shape(H, W)

        self.total_traffic = float(self.traffic_map.sum() + 1e-8)

    # ---------------------------
    # 상태 관련 유틸
    # ---------------------------
    def random_state(self) -> np.ndarray:
        """
        랜덤한 기지국 배치를 생성.
        반환: shape (n_bs, 2) (x, y)
        """
        xs = self.rng.integers(0, self.width, size=self.n_bs)
        ys = self.rng.integers(0, self.height, size=self.n_bs)
        state = np.stack([xs, ys], axis=1).astype(int)
        return state

    def clamp_state(self, state: np.ndarray) -> np.ndarray:
        """
        state 내 좌표를 격자 범위 안으로 강제(clamp)
        """
        s = state.copy()
        s[:, 0] = np.clip(s[:, 0], 0, self.width - 1)
        s[:, 1] = np.clip(s[:, 1], 0, self.height - 1)
        return s

    def get_neighbor(self, state: np.ndarray) -> np.ndarray:
        """
        현재 state에서 하나의 기지국 위치를 살짝 이동시킨 neighbor 생성
        - 1개의 BS를 골라 [-1, 0, +1] 범위에서 랜덤 이동
        """
        s = state.copy()
        idx = self.rng.integers(0, self.n_bs)
        dx = self.rng.integers(-1, 2)
        dy = self.rng.integers(-1, 2)
        s[idx, 0] += dx
        s[idx, 1] += dy
        s = self.clamp_state(s)
        return s

    # ---------------------------
    # 평가 함수
    # ---------------------------
    def evaluate(self, state: np.ndarray) -> tuple[float, dict]:
        """
        주어진 state에 대한 비용(cost)과 메트릭 반환
        반환: (cost, metrics_dict)
          - cost: 최소화 대상
          - metrics:
              'coverage_ratio': 커버된 트래픽 / 전체 트래픽
              'num_bs': 기지국 수
        """
        s = self.clamp_state(state)
        if s.shape[0] == 0:
            # 기지국이 하나도 없으면 최악의 cost
            return 1e9, {"coverage_ratio": 0.0, "num_bs": 0}

        # positions: shape (K, 2)
        xs = s[:, 0][:, None, None]  # (K, 1, 1)
        ys = s[:, 1][:, None, None]  # (K, 1, 1)

        dx2 = (xs - self.grid_x) ** 2
        dy2 = (ys - self.grid_y) ** 2
        dist2 = dx2 + dy2  # (K, H, W)

        min_dist2 = dist2.min(axis=0)  # (H, W)
        covered = min_dist2 <= self.r2

        covered_traffic = float((self.traffic_map * covered).sum())
        coverage_ratio = covered_traffic / self.total_traffic

        # 비용 함수: (1 - coverage_ratio)에 패널티
        # + 기지국 개수에 따른 패널티(지금은 w_num_bs=0이므로 영향 없음)
        uncovered_cost = (1.0 - coverage_ratio) * self.cost_weights.w_uncovered
        bs_cost = s.shape[0] * self.cost_weights.w_num_bs
        cost = uncovered_cost + bs_cost

        metrics = {
            "coverage_ratio": coverage_ratio,
            "num_bs": int(s.shape[0]),
            "uncovered_cost": float(uncovered_cost),
            "bs_cost": float(bs_cost),
        }
        return float(cost), metrics
